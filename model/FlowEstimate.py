'''
Source code for the time-continuous manifold sampling branch
'''
import torch
import torch.nn as nn
import numpy as np

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

def warp(Input, Flow): #B,C,H,W, B,2,H,W
    device = Flow.device
    Horizontal = torch.linspace(-1.0, 1.0, Flow.shape[3]).view(
        1, 1, 1, Flow.shape[3]).expand(Flow.shape[0], -1, Flow.shape[2], -1).to(device)
    Vertical = torch.linspace(-1.0, 1.0, Flow.shape[2]).view(
        1, 1, Flow.shape[2], 1).expand(Flow.shape[0], -1, -1, Flow.shape[3]).to(device)
    Grid = torch.cat([Horizontal, Vertical], 1).to(device)

    Flow = torch.cat([Flow[:, 0:1, :, :] / ((Input.shape[3] - 1.0) / 2.0),
                         Flow[:, 1:2, :, :] / ((Input.shape[2] - 1.0) / 2.0)], 1)

    g = (Grid + Flow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=Input, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.PixelShuffle(2)
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )
    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature = self.upsample(motion_feature) #/4 /2 /1

        if flow != None:
            x = torch.cat((x, flow), 1)
        x = self.conv(torch.cat([motion_feature, x], 1))

        flow = x[:, :4]
        mask = x[:, 4:5]
        return flow, mask


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1,
                                 bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, 1, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class InfofusionHead(nn.Module):
    def __init__(self, in_planes, c):
        super(InfofusionHead, self).__init__()
        self.upsample = nn.PixelShuffle(2)
        self.Fuse = nn.Sequential(
                                  conv(in_planes, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )
    def forward(self, envinfo, img, ev): # /16 /8 /4
        envinfo = self.upsample(envinfo) #/4 /2 /1
        if ev != None:
            img = torch.cat((img, ev), 1)
        img = self.Fuse(torch.cat([envinfo, img], 1))

        ev = img[:, :4]
        weight = img[:, 4:5]
        return ev, weight

class RefineNet(nn.Module):
    def __init__(self, cin, base_planes):
        super(RefineNet, self).__init__()
        self.upsample = nn.PixelShuffle(2)
        self.refine = nn.Sequential(
            conv(4 * base_planes + cin*5+1, 2 * base_planes),
            conv(2 * base_planes, cin))
    def forward(self, evinfo, iis, it_est):
        x = torch.cat([self.upsample(evinfo), iis, it_est], 1)
        x = self.refine(x)
        return torch.sigmoid(x)

class FlowEst(nn.Module):
    def __init__(self, stages, cin, emb_dim):
        super(FlowEst, self).__init__()
        self.stages = stages
        self.headlist = nn.ModuleList()
        cin = 4
        for i in range(stages):
            if i == stages-1:
                self.headlist.append(InfofusionHead(emb_dim + cin*2, emb_dim // 2))
            else:
                self.headlist.append(InfofusionHead(emb_dim + cin*4+5, emb_dim // 2))

        self.RefineNet = RefineNet(cin, emb_dim//4)

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(warp(x[:B], flow[:, 0:2]))
            y1.append(warp(x[B:], flow[:, 2:4]))
        return y0, y1

    def forward(self, i1, i2, ens, fls, frames):
        B = i1.size(0)
        ts = np.linspace(0,1,frames+2)[1:-1]
        frames = []
        for timestep in ts:
            for i in range(self.stages-1, -1, -1):
                if i == self.stages-1:
                    evinfo = torch.cat([fls[i][:B] * timestep, fls[i][B:] * (1 - timestep), ens[i][:B] * timestep, ens[i][B:] * (1 - timestep)],
                                       dim=1)  # fl0-t, flt-1, en0, en1
                    iis = torch.cat([i1, i2], 1)
                    ev, weight = self.headlist[i](evinfo, iis, None)
                else:
                    evinfo = torch.cat([fls[i][:B] * timestep, fls[i][B:] * (1 - timestep), ens[i][:B] * timestep, ens[i][B:] * (1 - timestep)],
                                       dim=1)  # fl0-t, flt-1, en0, en1
                    iis = torch.cat([i1, i2, i1_est, i2_est, weight], 1)
                    ev_d, weight_d = self.headlist[i](evinfo, iis, ev)
                    ev = ev + ev_d
                    weight = weight + weight_d
                i1_est = warp(i1, ev[:, :2])
                i2_est = warp(i2, ev[:, 2:])
            it_est = i1_est * (1 - torch.sigmoid(weight)) + i2_est * torch.sigmoid(weight)

            evinfo = torch.cat([fls[-1][:B] * timestep, fls[-1][B:] * (1 - timestep), ens[-1][:B] * timestep, ens[-1][B:] * (1 - timestep)],dim=1)  # fl0-t, flt-1, en0, en1
            iis = torch.cat([i1, i2, i1_est, i2_est, weight], 1)
            residual = self.RefineNet(evinfo, iis, it_est) * 2 - 1
            
            frames.append(it_est + residual)
        frames = torch.stack(frames, 1)
        return frames



if __name__ == '__main__':
    n = FlowEst(3, 256)
    img0 = torch.randn(2, 3, 64, 64)
    img1 = torch.randn(2, 3, 64, 64)
    ens = [torch.randn(4, 256, 32, 32), torch.randn(4, 256, 32, 32), torch.randn(4, 256, 32, 32)]
    fls = [torch.randn(4, 256, 32, 32), torch.randn(4, 256, 32, 32), torch.randn(4, 256, 32, 32)]
    timestep = 0.5

    c = n(img0, img1, ens, fls, timestep)



