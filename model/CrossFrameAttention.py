'''
in:  B * emb_dim * H * W                         #F_Img1
     B * emb_dim * H * W                         #F_Img2
out: (
     ens: [(2*B) * emb_dim * H * W] * blocks     #[:B,:,:,:] -- En_Img1  [B:,:,:,:] -- En_Img2
     fls: [(2*B) * emb_dim * H * W] * blocks     #[:B,:,:,:] -- Fl_Img1  [B:,:,:,:] -- Fl_Img2
)
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def unpatchembbed(x, embed_dim, x_size):
    B, HW, C = x.shape
    x = x.transpose(1, 2).view(B, embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
    return x

def patchembbed(x):
    x = x.flatten(2).transpose(1, 2)
    return x

def get_cor(shape, device):
    tenHorizontal = torch.linspace(-1.0, 1.0, shape[2]).view(
        1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).to(device)
    tenVertical = torch.linspace(-1.0, 1.0, shape[1]).view(
        1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).to(device)
    cor = torch.cat([tenHorizontal, tenVertical], 1).to(device)
    return cor

def singletimeinput(time, dim = 64):
    time0 = time[0]
    y0 = time0[0]
    m0 = time0[1]
    d0 = time0[2]
    h0 = time0[3]
    mi0 = time0[4]
    time1 = time[1]
    y1 = time1[0]
    m1 = time1[1]
    d1 = time1[2]
    h1 = time1[3]
    mi1 = time1[4]

    dm = m1-m0 if m1>=m0 else 12+m1-m0
    t0 = datetime(y0, m0, d0, h0, mi0)
    t1 = datetime(y1, m1, d1, h1, mi1)
    delta = abs(t1-t0)
    dd = delta.days
    dh,remainder = divmod(delta.seconds, 3600)
    dh = dh + dd*24
    dmin, _ = divmod(remainder, 60)
    dmin = dmin + dh * 60
    ddtrue = (datetime(y1, m1, d1, 0, 0) - datetime(y0, m0, d0, 0, 0)).days
    if m0 == 12:
        daysinm = 31
    else:
        daysinm = (datetime(y0, m0 + 1, 1, 0, 0) - datetime(y0, m0, 1, 0, 0)).days

    mx = np.linspace(m0,m0+dm,dim)/12*2*np.pi
    mt1 = torch.tensor(np.sin(mx))
    mt2 = torch.tensor(np.cos(mx))

    dx = np.linspace(d0, d0+ddtrue,dim)/daysinm*2*np.pi
    dt1 = torch.tensor(np.sin(dx))
    dt2 = torch.tensor(np.cos(dx))

    hx = np.linspace(h0, h0+dh,dim)/24*2*np.pi
    ht1 = torch.tensor(np.sin(hx))
    ht2 = torch.tensor(np.cos(hx))
    return torch.stack([mt1, mt2, dt1, dt2, ht1, ht2])

def gettimeinput(times, dim=64):
    tts = []
    for i in range(times.size()[0]):
        tts.append(singletimeinput(times[i], dim))
    return torch.stack(tts, 0).permute(0,2,1)

class TimeAttentionCal(nn.Module):
    def __init__(self, dim, length, num_heads, dropout):
        super(TimeAttentionCal, self).__init__()
        self.proj_q = nn.Linear(6, dim)
        self.proj_kv = nn.Linear(dim, dim * 2)
        self.drop = nn.Dropout(dropout)
        self.dim = dim
        self.length = length
        self.n_heads = num_heads
        self.scores = None
    def forward(self, i, t):  # i: B, L, emb_dim, i2: B, [[y0,m0,d0,t0,mi0],[y1,m1,d1,t1,mi1]]
        t = gettimeinput(t, self.length).float().to(i.device)
        t_ = t.flip(1)
        q = self.proj_q(torch.cat([t, t_], dim=0))
        k, v = self.proj_kv(i).chunk(2, dim=-1)  # B, H*W, emb_dim,
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        return h

class CrossFrameAttentionCal(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super(CrossFrameAttentionCal, self).__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_kv = nn.Linear(dim, dim * 2)
        self.proj_mo = nn.Linear(2, dim)
        self.drop = nn.Dropout(dropout)
        self.dim = dim
        self.n_heads = num_heads
        self.scores = None
    def forward(self, i1, i2, cr):  # i1: B, L, emb_dim, i2: B, L, emb_dim
        q = self.proj_q(i1)
        k, v = self.proj_kv(i2).chunk(2, dim=-1)  # B, H*W, emb_dim,
        mo = self.proj_mo(cr)
        q, k, v, mo = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v, mo])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)

        mo = (scores @ mo).transpose(1, 2).contiguous()
        mo = merge_last(mo, 2)
        return h, mo
class CrossModelAttentionCal(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super(CrossModelAttentionCal, self).__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_kv = nn.Linear(dim, dim * 2)
        self.drop = nn.Dropout(dropout)
        self.dim = dim
        self.n_heads = num_heads
        self.scores = None
    def forward(self, i1, i2):  # i1: B, L, emb_dim, i2: B, L, emb_dim
        q = self.proj_q(i1)
        k, v = self.proj_kv(i2).chunk(2, dim=-1)  # B, H*W, emb_dim,
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        return h

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout2:
            out = self.dropout2(out)
        return out

class CrossFrameAttentionLayer(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1,
                 attn_dropout_rate=0.1):
        super(CrossFrameAttentionLayer, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.attn = CrossFrameAttentionCal(in_dim, num_heads, attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm3 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)
    def forward(self, x1, x2, cr):
        residual = x1  # B, H*W, emb_dim
        out, mo = self.attn(self.norm1(x1), self.norm2(x2), cr)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out
        out = self.norm3(out)
        out = self.mlp(out)
        out += residual
        return out, mo
class CrossModelAttentionLayer(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1,
                 attn_dropout_rate=0.1):
        super(CrossModelAttentionLayer, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.attn = CrossModelAttentionCal(in_dim, num_heads, attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm3 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)
    def forward(self, x1, x2):
        residual = x1  # B, H*W, emb_dim
        out = self.attn(self.norm1(x1), self.norm2(x2))
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out
        out = self.norm3(out)
        out = self.mlp(out)
        out += residual
        return out
class TimeAttentionLayer(nn.Module):
    def __init__(self, in_dim, mlp_dim, in_length, num_heads, dropout_rate=0.1,
                 attn_dropout_rate=0.1):
        super(TimeAttentionLayer, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = TimeAttentionCal(in_dim, in_length, num_heads, attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm3 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)
    def forward(self, x, t):
        residual = x  # B, H*W, emb_dim
        out = self.attn(self.norm1(x), t)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out
        out = self.norm3(out)
        out = self.mlp(out)
        out += residual
        return out

class MultiModeFusionBlock(nn.Module):
    def __init__(self, emb_dim, mlp_dim, x_size, num_layers=4, num_heads=4,
                 dropout_rate=0.1,
                 attn_dropout_rate=0.0):
        super(MultiModeFusionBlock, self).__init__()
        self.emb_dim = emb_dim
        self.x_size = x_size
        self.CFA_layers = nn.ModuleList()
        for i in range(num_layers):
            self.CFA_layers.append(CrossModelAttentionLayer(emb_dim, mlp_dim,num_heads, dropout_rate, attn_dropout_rate))
    def forward(self, m, f): #B,2,H,W
        m, f = patchembbed(m), patchembbed(f)
        for layer in self.CFA_layers:
            m = layer(m, f)
        return unpatchembbed(m, self.emb_dim, self.x_size)

class CrossFrameAttentionBlock(nn.Module):
    def __init__(self, emb_dim, mlp_dim, x_size, num_layers=4, num_heads=4,
                 dropout_rate=0.1,
                 attn_dropout_rate=0.0):
        super(CrossFrameAttentionBlock, self).__init__()
        self.emb_dim = emb_dim
        self.x_size = x_size
        self.TA_layer = TimeAttentionLayer(emb_dim, mlp_dim, x_size[0]*x_size[1], num_heads, dropout_rate=0.1,attn_dropout_rate=0.1)
        self.CFA_layers = nn.ModuleList()
        for i in range(num_layers):
            self.CFA_layers.append(CrossFrameAttentionLayer(emb_dim, mlp_dim,num_heads, dropout_rate, attn_dropout_rate))
    def forward(self, i, t, cr, B): #B,2,H,W
        i1 = patchembbed(i)
        i1 = self.TA_layer(i1, t)
        i2 = torch.cat([i1[B:], i1[:B]])
        cr = patchembbed(cr)

        for layer in self.CFA_layers:
            i1, mo = layer(i1, i2, cr)
        return unpatchembbed(i1, self.emb_dim, self.x_size), unpatchembbed(mo, self.emb_dim, self.x_size)

class CrossFrameAttentionSubNet(nn.Module):
    def __init__(self, blocks, layers, emb_dim, mlp_dim, x_size, num_heads=4, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(CrossFrameAttentionSubNet, self).__init__()
        self.CrossFrameAttnblocks = nn.ModuleList()
        self.MultiModeAttnblocks = nn.ModuleList()
        self.blocks = blocks
        for i in range(blocks):
            self.CrossFrameAttnblocks.append(CrossFrameAttentionBlock(emb_dim, mlp_dim, x_size, layers, num_heads, dropout_rate, attn_dropout_rate))
            self.MultiModeAttnblocks.append(MultiModeFusionBlock(emb_dim, mlp_dim, x_size, 2, num_heads, attn_dropout_rate))
    def forward(self, i1, i2, time, fy1, fy2):
        B, _, H, W = i1.shape
        i = torch.cat([i1, i2], dim = 0)
        shape = [2 * B, H, W]
        cr = get_cor(shape, i1.device)
        # i1 = patchembbed(i1)
        # i2 = patchembbed(i2)
        # cr = patchembbed(cr)
        ens = []
        fls = []
        for block in range(self.blocks):
            i1, i2 = i.chunk(2, dim = 0)
            i1 = self.MultiModeAttnblocks[block](i1, fy1)
            i2 = self.MultiModeAttnblocks[block](i2, fy2)
            i = torch.cat([i1, i2], dim = 0)
            i, fl = self.CrossFrameAttnblocks[block](i, time, cr, B) #i:  i1B+i2B
            ens.append(i)
            fls.append(fl)
        return ens, fls