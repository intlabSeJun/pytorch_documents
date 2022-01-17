import torch
import torch.nn as nn


# Conv1D
"""
cross-correlation 적용.
보통 sequential data에 사용 됨.
이미지에서의 텐서와 다른 구조로 생각하면 편함.
- 이미지 : (N,C,H,W)
- 자연어 : (N,C,vector) # 여기서 C는 time-step
  - vector가 C개의 channel을 가지고 있는 구조로 생각하면 편하다.
Conv1d 진행은 각 channel에 각 vecotr에서 filter을 sliding 통해 연산하고 합한다. 
"""
"""
m = nn.Conv1d(16,33,3,stride=2)  #(in_channel, out_channel, kernel_size)
input = torch.randn(20,16,50) # dim=50 vector가 time-step=16만큼의 구조가 batch=20만큼 있다.
out = m(input) # 3x3 filter가 16 in_channel, 33 out_channel 만큼 있고, 
               # input의 각 channel마다 conv연산을 수행함. (20,33,24) 됨.
print(out.shape)#"""

#머신러닝에서 convolution은 cross-correlation이다.
"""
https://mcheleplant.blogspot.com/2019/08/convolution-vs-cross-correlation.html
- convolution : 연산시 함수일 경우 y축 대칭, 행렬 곱일 경우 가로,세로 대칭(대각선 대칭)하여 연산수행
- cross-correlation : 대칭 없이 연산 수행.
- 딥러닝 프레임워크에서는 모두 cross-correlation 수행. 
  - 통상적으로 Convoluion 이라고 부름, 이는 대칭 해도 실제 연산결과와 크게 차이가 없기 떄문.
"""

#Convolution vs cross-correlation
"""
https://figureking.tistory.com/177
- convolution은 연산 작용.
  - 원래의 변수가 출력에도 그대로 살아 있다?
  - 결과가 자신이 속해 있는 공간 그 자체로 보내짐.
- correlation은 변환 작용.
  - 원래의 변수가 다른 변수로 바꾸이 출력에 나타남.
  - 자신이 속한 공간이 아닌, 타 공간에 보내짐. 
  - 변환의 일종, 표현 영역을 바꾸어 다르게 표현함으로써 해석 용이, 취급 단순화 등.
-> 정리하면 딥러닝에서 conv 사용하는 이유는 변환의 일종으로 볼 수 있고, 이는 수학적으로는 correlation이지만 conv라고 불림.
"""
