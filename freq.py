'''
Author: your name
Date: 2020-12-03 22:24:19
LastEditTime: 2020-12-06 22:14:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /fast-reid/freq.py
'''
import torch
from torch import nn
import math

class FreqAttentionLayerSE(torch.nn.Module):
    def __init__(self, channel, fea_h, fea_w, reduction = 16):
        super(FreqAttentionLayerSE, self).__init__()
        self.reduction = reduction

        # mapper_x = [0,0,1,-1,0,3,0,2]
        # mapper_y = [0,1,0,-1,3,0,2,0]
        mapper_x = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,5,5,5,5]
        mapper_y = [0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,0,1,2,3]
        assert len(mapper_x) == len(mapper_y) == 32
        print('32 freqs')

        self.dct_layer = DctMul(fea_h, fea_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape

        y = self.dct_layer(x)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)



class DctMul(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, width, height, mapper_x, mapper_y, channel):
        super(DctMul, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        # num_freq, h, w

    def forwardv1(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        n, c, h, w = x.shape
        x = x.contiguous().view(n * c, 1, h, w)

        result = torch.sum(x * self.weight, dim=[2,3])
        return result

    def forwardv2(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result
    
    def forward(self,x):
        return self.forwardv2(x)
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_fileter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_fileter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_fileter

if __name__ == '__main__':
    a  = FreqAttentionLayerSE(channel=2048, fea_h=7, fea_w=7)
    pass
