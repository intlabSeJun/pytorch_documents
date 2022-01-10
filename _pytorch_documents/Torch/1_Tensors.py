import torch

#tensor
"""
x = torch.tensor([1,2,3])
y = x
print(torch.is_tensor(x)) # torch 여부 확인#"""

# Storage ( tensor 안에 들어가는 데이터 구조. 딥러닝에는 직접적으로 쓸 일 없다고 함. )
"""
B = torch.DoubleStorage(5)
print(B)
print(torch.is_storage(B))#"""

# complex, 복소수
# real / imag 구성. (실수/허수)
# 입력이 torch.float32 -> torch.complex64, torch.float64 -> torch.complex128
"""
real = torch.tensor([1,2], dtype=torch.float32)
imag = torch.tensor([3,4], dtype=torch.float32)
z = torch.complex(real, imag) #tensor([1.+3.j, 2.+4.j])
print(z)
print(z.dtype)
# 복소수를 사용하는 이유 https://blog.naver.com/mykepzzang/222203363902
# -> 숫자의 표현 범위를 사용하기 위해, 보통 물리에서 많이 사용됨.(운동량을 계산하는 경우)
# -> 딥러닝에서도 사용 가능할듯?#"""

# conjugate
# 허수의 부호를 바꾼다. 복소평면에서 x축 대칭하는 것.
# 복소수가 아니면 수행 안됨(에러안남)
"""
x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
print(x)
print(x.is_conj()) #False
y = torch.conj(x)
print(y)
print(y.is_conj()) #True
z = torch.conj(y)
print(z)
# 복소수 아닌 경우
b = torch.tensor([1,2,3])
c = torch.conj(b)
print(f'b:{c, c.is_conj()}')#"""


#floating_point(부동소수점) & float dtype
"""
● float16, float32, float64 & floating point 고찰.
딥러닝에서 이러한 dtype의 차이는 무엇일까?
-결론적으로 딥러닝에서 float32를 많이 사용하나, float16 사용시 정확도 문제로 인해 두 개를 혼합한 것도 사용함."""
"""
● float16 vs float32 in 딥러닝.
https://stackoverflow.com/questions/46613748/float16-vs-float32-for-convolutional-neural-networks
- 차이는 크게 없지만 float16을 사용할 경우 메모리를 줄일 수 있고, 빠르다. 
- 하지만 loss scale을 너무 크게 할 경우 NAN, IFN 발생할 수 있으며, 너무 작게 할 경우 수렴이 안됨.
 ( 가끔씩 실험중에 NAN 뜨는 문제를 flaot dtype도 확인하면 되겠네 )#"""
"""
● floating point (부동소수점)
https://ko.wikipedia.org/wiki/%EB%B6%80%EB%8F%99%EC%86%8C%EC%88%98%EC%A0%90
- 컴퓨터는 이진코드로 숫자를 저장하기 때문에 저장 방식이 다른데 torch는 standard IEEE 부동소수점 방식을 사용.
- standard IEEE 부동소수점
  : 부호, 지수, 가수 부분으로 나누어 지며, float32, float64, float16에 따라 공간이 다르다.
  + what is different single precision and double precision & multi-precision, mix-precision
   https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multi-and-mixed-precision-computing/
   float 64 - double precision
   float 32 - single precision
   float 16 - half precision
   ○ multi-precision, mix-precision
    ▷ multi-precision
     https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multi-and-mixed-precision-computing/
     - 이건 그냥 경우에 따라서 맞게 32 or 16 or 8 쓰는 거네;
    ▷ mix-precision
      https://bo-10000.tistory.com/32
      - 문제 : float32를 사용하면 메모리를 많이 먹으나 정밀도가 좋아지고, float16을 사용하면 메모리를 적게 먹으나 정밀도가 떨어짐.
      - 이러한 문제를 해결하기 위해 두 개를 혼합해서 사용하는 논문이 있음.
      - FP32로 학습된 gradient 셈플링 결과 0에 치우치는 것들이 보임
      - 따라서 0에 해당하는 gradient는 버리면서 메모리를 줄이도록 하고자 함.
      - 방법 : FP32 wieght는 계속 저장해 두고, FP16 copy weight를 만들어 forward,backward pass 진행
              FP16 copy weight으로 얻은 gradient를 이용해 FP32 weight 업데이트. 
             -> 기존 32비트 weight를 16비트로 만듬으로써 매우 작은 값들은 버리고 진행하고,
                16비트 gradient를 통해 32비트 weight를 업데이트 한다. 
             + scale factor 설정 해줘야 함.(경험적으로, 논문 참고)
      ▷ how to use mix-precision in Pytorch
      nvidia 공식 : https://developer.nvidia.com/automatic-mixed-precision
      한글 블로그 : https://blog.naver.com/PostView.naver?blogId=ehdrndd&logNo=222506364024&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
      
 ○ bfloat16 vs float16
https://stackoverflow.com/questions/44873802/what-is-tf-bfloat16-truncated-16-bit-floating-point
- bfloat16 : float32에서 지수 부분을 8비트, 가수 부분을 7비트로 자른 것. 쉽게 float32로 변환이 가능함.
             -> NaN값이 적게 나오게 하며, exploding/vanishing gradients 최소화
   
"""
"""
x = torch.tensor([0.1,0.3]) #기본 float 32
print(x.dtype)
x = torch.tensor([0.1,0.3], dtype=float) #float64
print(x.dtype)
# float 바꾸는 여러 방법들
print(x.half().dtype) # 어떤 dtype이든 16으로 바꿈.
torch.set_default_dtype(torch.float16) # 모든 tensor 생성시 기본 16비트
x = torch.tensor([0.1,0.3])
print(x.dtype)#"""

#torhc.is_nonzero
# - zero인지 확인.
"""
print(torch.is_nonzero(torch.tensor([0.]))) # False
#"""

#torch.get_default_dtype
# - 현재 기본 설정된 dtype 알 수 있다.
"""
print(torch.get_default_dtype()) # torch.float32
torch.set_default_dtype(torch.float64)
print(torch.get_default_dtype()) # torch.float64
torch.set_default_tensor_type(torch.FloatTensor) # setting tensor type
print(torch.get_default_dtype())#"""

#torch.set_default_tensor_type
# - set torch tensor dtype
"""
what is differnet torch.dtype and torch.tensortype
https://discuss.pytorch.org/t/difference-between-torch-dtype-and-torch-tensortype/58948/2
아래는 같은 코드
float_tensor = torch.tensor([4,5,6], dtype=torch.float)
float_tensor = torch.FloatTensor([4,5,6])#"""
"""
print(torch.tensor([1.,2.,3.]).dtype) #tensor type float32
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.,2.,3.]).dtype) # tensor type float64"""

# torch.numel
# - tensor elements 수.
"""
a = torch.randn(1,2,3,4,5)
print(torch.numel(a))  #120 = 1*2*3*4*5
a = torch.randn(1,2,3,4,5).cuda() # cuda 넣어도 요소 개수만 출력됨.
print(torch.numel(a), a)  #120 = 1*2*3*4*5#"""

#torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
# - 출력을 여러 옵션을 통해..
""" argument
 - precision : default=4, 출력할 소수점 자리수.
 - threshold : default=1000.(float), 모르겠음.@@
 - profile : "full"-전체 tensor 출력, "default"-설정한 option 모두 초기화
 - linewidth : 라인 간격인데 default=80이고 더 줄이면 세로로 나오고, 늘리면 효과 없음.
 - edgeitems : default=3, tensor가 많을 때 보여줄 개수(행,열 모두)
 - sci_mode : default=Ture, notation on/off, 표현방식. 
"""
"""
# 예시 https://stackoverflow.com/questions/52673610/printing-all-the-contents-of-a-tensor
x = torch.rand(2,3)
print(x)
torch.set_printoptions(precision=2)
print(x)
torch.set_printoptions(profile="default") #reset
#torch.set_printoptions(threshold=1)
x = torch.rand(100,100)
torch.set_printoptions(edgeitems=5) # tensor 수가 많을 때, 행,열 보여줄 개수.
print(x)
torch.set_printoptions(profile="default")
print(x)
torch.set_printoptions(linewidth=2000) #간격을 말하는 거 같은데 거의 차이 없는 듯.
print(x)
torch.set_printoptions(profile="default")
torch.set_printoptions(profile="full") #전체 tensor 출력.
print(x)
x = torch.tensor([0.000000234])
print(x)
torch.set_printoptions(sci_mode=False) # notation on/off, 표현방식. default=True
print(x)
torch.set_printoptions(profile="default")#"""


















