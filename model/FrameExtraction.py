'''
in:  B * Cin * H * W                        #Img
out: B * (4*emb_dim) * (H/2) * (W/2)        #F_Img
'''
import torch
import torch.nn as nn
import pywt
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def batch_wavelet_decomposition(batch_array, wavelet='haar'):
    B, C, H, W = batch_array.shape
    batch_array = batch_array.cpu()
    new_H, new_W = H // 2, W // 2
    decomposed = np.zeros((B, C * 4, new_H, new_W))

    for b in range(B):
        for c in range(C):
            img = batch_array[b, c]
            LL, (LH, HL, HH) = pywt.dwt2(img, wavelet)
            # 将4个系数堆叠到通道维度
            decomposed[b, c * 4] = LL
            decomposed[b, c * 4 + 1] = LH
            decomposed[b, c * 4 + 2] = HL
            decomposed[b, c * 4 + 3] = HH
    return decomposed

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

class WaveletExtraction(nn.Module):
    def __init__(self, Cin, emb_dim):
        super(WaveletExtraction, self).__init__()
        self.branch1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4*Cin, out_channels=emb_dim//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(emb_dim//2),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16*Cin, out_channels=emb_dim//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(emb_dim//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=emb_dim//2, out_channels=emb_dim//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(emb_dim//2),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        device = x.device
        x_l1 = torch.from_numpy(batch_wavelet_decomposition(x, 'db1')).float().to(device)
        out = self.branch1(x_l1)
        x_l2 = torch.from_numpy(batch_wavelet_decomposition(x_l1, 'db1')).float().to(device)
        out2 = self.branch2(x_l2)
        return torch.cat([out, out2],1)

class Res2NetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False,  norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = groups * planes
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class FrameExtraction(nn.Module):
    def __init__(self, Cin = 3, planes = 256, layers = [4,4,4], zero_init_residual=False,
                 groups=1, scales=1, se=False, norm_layer=None):
        super(FrameExtraction, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        planes = [int(planes * 2 **i) for i in range(3)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(Cin, planes[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)

        self.wave = WaveletExtraction(Cin, planes[0])
        self.fusewx = nn.Conv2d(planes[0]*2, planes[0], kernel_size=3, stride=1, padding=1)
        self.fusewx2 = nn.Conv2d(planes[0]*2, planes[0], kernel_size=3, stride=1, padding=1)
        
        self.layer1 = self._make_layer(Res2NetBottleneck, planes[0], layers[0], scales=1, groups=groups, se=se,
                                       norm_layer=norm_layer)
        self.layer2 = self._make_layer(Res2NetBottleneck, planes[1], layers[1], stride=1, scales=scales, groups=groups,
                                       se=se, norm_layer=norm_layer)
        self.layer3 = self._make_layer(Res2NetBottleneck, planes[2], layers[2], stride=2, scales=scales, groups=groups,
                                       se=se, norm_layer=norm_layer)


        self.fuse1 = nn.Conv2d(planes[0], planes[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.fuse2 = nn.Conv2d(planes[1], planes[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.fuse3 = nn.Conv2d(planes[2], planes[2], kernel_size=3, stride=1, padding=1, bias=False)

        self.MultiScaleProj = nn.Sequential(
            CALayer(planes[0]+planes[1]+planes[2]),
            norm_layer(planes[0]+planes[1]+planes[2]),
            nn.Conv2d(planes[0]+planes[1]+planes[2], planes[2], 3, 1, 1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Res2NetBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, scales=1, groups=1, se=False, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample, stride=stride, scales=scales, groups=groups, se=se, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, scales=scales, groups=groups, se=se, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        xw = self.wave(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fusewx(torch.cat((x, xw), dim=1))
        
        x1 = self.layer1(x)
        x1 = self.fusewx2(torch.cat((x1, xw), dim=1))
        
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x1 = self.fuse1(x1)
        x2 = self.fuse2(x2)
        x3 = self.fuse3(x3)

        return self.MultiScaleProj(torch.cat([x1, x2, x3], 1))
