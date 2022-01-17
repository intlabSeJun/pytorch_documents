import torch
import torch.nn as nn


# conv2d
"""
https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
- 일반적인 conv
- 대각선 conv (stride = (3,1) 등 사용)
- Atrous conv (dilation 사용)
- grou conv, depth-wise conv (groups 사용)
"""
# 일반적인 conv
"""
m = nn.Conv2d(16,33,3,stride=2)

# 대각선 conv. filter가 stride의 따라서 대각선으로 연산할 수 있음.
m2 = nn.Conv2d(16,33,(3,5), stride=(2,1), padding=(4,2))

# Atrous conv. filter가 연산시에 지정된 (height, width)따라 간격을 벌려 연산. 자세한건 사이트 참고.
m3 = nn.Conv2d(3,2,(3,5), stride=1, dilation=(3,1)) #"""

# grou conv, depth-wise conv (groups 사용) https://supermemi.tistory.com/116
"""
group conv  
https://supermemi.tistory.com/116
: conv2d에서 in_channel을 gropus 수로 나눌 수 있어야 함.
  기존과 다르게 input의 모든 feature-map을 사용하여 conv 연산이 되는게 아니라 그룹별로 분할되서 연산을 수행.
  
depth-wise conv  
https://utto.tistory.com/2
 : conv2d에서 groups 수를 in_channel과 동일시하면 됨. 
  채널 방향 연산x, 공간적인 연산, input channel 각각에 filter 수행하고 concat
  기존보다 연산량 감소, 기존에는 채널별로 conv한 것을 합산하고 이를 out_channel수만큼 반복하지만 depth-wise에서는 각 채널마다 필터 연산하고 concat 하므로.#"""
"""
m = nn.Conv2d(4,2,(3,5), stride=1, groups=2)
print(m.weight.shape) # (2,2,3,5) 형태. 그룹으로 나눈듯.
input = torch.randn(1,4,10,10)
out = m(input)
print(out.shape) #(1,2,8,6)#"""

