import numpy
import torch

# torch.tensor(), clone(), detach(), requires_grad, 주소 체크.
# 결론 : 주소가 안겹치게 사용 하려면 detach()
"""
a = [2,3,4]
print(f'a:{a}, id(a):{id(a)}')
b = torch.tensor(a) # 새롭게 주소 할당. tensor 주소와 data 주소가 다르다.
print(f'b=torch.tensor(a)\nid(b):{id(b)}, id(b.data): {id(b.data)}')
c = b.clone() # b.data가 tensor 주소가 되고, c.data 주소는 새로 할당.
print(f'c=b.clone()\nid(c): {id(c)}, id(c.data): {id(c.data)}')
d = b # tensor 주소만 같고, data는 새로 할당.
print(f'd=b\nid(d): {id(d)}, id(d.data): {id(d.data)}')
e = b.detach() # tensor, data 모두 새로 할당.
print(f'e=b.detach()\nid(e): {id(e)}, id(e.data): {id(e.data)}')
kk = [0.2, 0.3, 0.4]
print(id(kk))
f = torch.tensor(kk, requires_grad=True) # tensor, data 모두 새로 할당.
print(f'f=torch.tensor(a, requires_grad=True)\nid(f): {id(f)}, f.data: {id(f.data)}')#"""

# torch.Tensor vs torch.tensor
# gpu 사용시 왠만하면 torch.tensor 사용하자.
"""
a = torch.Tensor([1]) # 무조건 float32, requires_grad=True 안됨
print(a, type(a), a.dtype)
a = torch.Tensor() #빈 클래스 가능
print(a, type(a), a.dtype)
b = torch.tensor([1.0], dtype=torch.float32, requires_grad=True) #자료형 지정 가능.
print(b, b.dtype)

# torch.as_tensor
# - 받은 data를 torch.Tensor에 넣어줌.
a = numpy.array([1, 2, 3])
t = torch.as_tensor(a)
print(t)#"""

#torch.as_strided(input, size, stride, storage_offset)
# - input을 view 하여 size만큼의 tensor로 stride를 통해 나타냄
#   stride : (a,b) -  a에서 시작하여 b만큼 떨어진 값을 나타냄(a는 두번째부터 반영)
    # 아래 예시 참고.
# storage_offset : 맨 처음 시작 위치.
"""
x = torch.randn(3,3)
print(x)
t = torch.as_strided(x, (2,2), (2,3)) # 처음에는 0번과 3번째 값을 2개 shape으로 나타냄 #두번쨰에는 2번 값과 거기서 2번 떨어진 3번 값을 나타냄
print(t)#"""

# torch.from_numpy()
# 메모리를 공유하면서 tesnor로 바꿈. 텐서 data 수정시 array에도 반영 됨.
# 주소와 메모리는 다른가본데? 주소가 같으면 메모리 공유 인줄 알았는데 확인 필요
"""
a = numpy.array([1,2,3])
print(a, a.dtype, id(a), id(a.data))
b = torch.from_numpy(a)
print(b.data_ptr(), b.data.data_ptr())
print(b, b.type, id(b),id(b.data)) #주소와 메모리는 다른가본데? 주소가 같으면 메모리 공유 인줄 알았는데 확인 필요;@
b[0] = -3
print(a, id(a)) #"""
# 메모리를 따로 할당하지 않는 Tensor 객체 연산
""" narrow(), view(), expand(), transpose(), permute(), ..
   원본 데이터를 따로 메모리에 할당하여 복사하지 않고 원본 메모리 주소를 공유함. 
   따라서 위 연산을 거친 Tensor 객체 내 데이터를 변경하면 원본도 함께 변경됨. ==같은 포인터 공유
   https://f-future.tistory.com/entry/Pytorch-Contiguous"""

#torch.frombuffer
