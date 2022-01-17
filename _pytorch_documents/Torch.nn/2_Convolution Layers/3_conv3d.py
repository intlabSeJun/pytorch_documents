import torch
import torch.nn as nn


# conv3d
"""
conv2d : 각 frame마다 고려(공간적)
conv3d : 여러 frame 고려(시·공간적), temporal information, video 사용
 - depth 추가 됨.
input dim : (batch, in_channel, depth, height, width)
3d cov dim : (in_channel, out_channel, kernel_size, stride, padding, …)
 - kernel_size가 depth도 결정함.
out dim : (batch, out_channels, out_depth, out_height, out_width)
"""
"""
m = nn.Conv3d(3,2,2, stride=2) # filter dim = (3,2,2,2,2)
print(m)
input = torch.randn(1, 3, 6, 8, 8)
out = m(input)
print(out.shape) #(1,2,3,4,4)#"""