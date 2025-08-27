from .FrameExtraction import FrameExtraction
from .CrossFrameAttention import CrossFrameAttentionSubNet, MultiModeFusionBlock
from .FlowEstimate import FlowEst
import torch
import torch.nn as nn
import math

        
class network(nn.Module):
    def __init__(self, Cin, emb_dim, blocks, attn_layer):
        super().__init__()
        self.FE_field = FrameExtraction(Cin, emb_dim) #multiscale deep-wavelet feature extraction branch
        self.FE_fy = FrameExtraction(6, emb_dim) #multiscale deep-wavelet feature extraction branch
        self.CFA = CrossFrameAttentionSubNet(blocks, attn_layer, emb_dim * 4, emb_dim * 8, (32, 32), 8) #cross-modal spatiotemporal information fusion branch
        self.ES = FlowEst(blocks,Cin,emb_dim*4) #time-continuous manifold sampling branch
    def forward(self, i1, i2, time=None, fy=None, outputframes = 4):
        fy_1, fy_2 = fy.chunk(2, 1)
        fy_1, fy_2 = fy_1.squeeze(1), fy_2.squeeze(1)

        f1_field = self.FE_field(i1)
        f2_field = self.FE_field(i2)

        f1_fy = self.FE_fy(fy_1)
        f2_fy = self.FE_fy(fy_2)

        f_fb, l_fb = self.CFA(f1_field, f2_field, time, f1_fy, f2_fy)
        out = self.ES(i1, i2, f_fb, l_fb, outputframes)

        return out

if __name__ == '__main__':
        time = torch.tensor([[[2024,   12,   30,    4,    0],
             [2024,   12,   30,   10,    0]],

            [[2024,   12,   31,    0,    0],
             [2024,   12,   31,    6,    0]],

            [[2024,   12,   31,    1,    0],
             [2024,   12,   31,    7,    0]],

            [[2024,   12,   31,    2,    0],
             [2024,   12,   31,    8,    0]],

            [[2024,   12,   31,    3,    0],
             [2024,   12,   31,    9,    0]],

            [[2024,   12,   31,    4,    0],
             [2024,   12,   31,   10,    0]]])
    n = network(3,96,2,2)
    c = n(torch.randn(6,3,64,64), torch.randn(6,3,64,64), time, torch.randn(6,2,6,64,64))
