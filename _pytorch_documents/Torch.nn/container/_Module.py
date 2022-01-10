import torch
import torch.nn as nn
import torch.nn.functional as F


# MODULE
"""
class Model(nn.Module):
    def __init__(self):
        #why use super().__init__()
        #https://daebaq27.tistory.com/60
        #super().__init__() #부모 클래스 초기화하여 클래스 속성을 subclass가 받아오도록 함. 하지 않으면 사용할 수 없다.
        
        super(Model, self).__init__() # 위와 같지만 self를 해줌으로써 현재 클래스가 어떤 클래스인지 명확히 표시
        # 사용하지 않으면 nn.Module 내부에서 __setattr__ 사용할 수 없어 extend 할 수 없음.
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

# add_module() : nn.Module에 상속받은 모델에 추가함.
# https://greeksharifa.github.io/pytorch/2018/11/10/pytorch-usage-03-How-to-Use-PyTorch/
model = Model()
last_module = nn.Conv2d(20,40,5)
model.add_module('last_module', last_module)
print(model)

# apply(fn) : 현재 모듈의 모든 submodule에 해당 함수(fn)을 적용. 주로 파라미터 초기화 시에 사용.
def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()
    elif isinstance(submodule, nn.Linear):
        submodule.weight.data.fill_(1.1)
        submodule.bias.data.fill_(0.5)

# 각 layer 유형에 맞게 파라미터 초기화.
model.apply(weight_init_xavier_uniform)#"""

# 모델에서 쌓은 layer가 무엇이고, 각 값들에 대해서 뽑을 수 있을까?
# 모델의 각 layer 파라미터 뽑는 generator
# naemd가 붙으면 (name, parameters(or children)), 붙지 않으면 parameters(or children) return
"""
for m in model.parameters():
    print(m)

# layer 이름과 파라미터 값들 출력
for m in model.named_parameters():
    print(m)

# layer 이름과 파라미터 따로 출력.
for name, p in model.named_parameters():
    print(name)

# layer 모듈 출력.
for name in model.named_modules():
    print(name)

# 이름 제외하고 layer 형식만
for k in model.children():
    print(k)

# get_parameters('name')
for name, p in model.named_parameters():
    print(name)
    print(model.get_parameter(name))

# modules()
for idx, m in enumerate(model.modules()):
    print(idx, '->', m)#"""


# buffer
"""
https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/8
https://teamdable.github.io/techblog/PyTorch-Module
Buffer는 Module에 저장해 놓고 사용하는 Tensor의 일종으로 학습을 통해 계산되지 않는 Tensor. 
requires_grad=False, cuda 사용 가능.
특정 중간 layer 값들을 변경하고 싶지 않거나, gradient 갱신을 하고 싶지 않은 경우에 사용. 
1. 일반적인 buffer
: torch.tensor()로 구성, state_dict으로 저장 못함, 따로 설정해줘야 함
2. register_buffer
: stated_dict으로 편리하게 저장 가능, 조건문에 따라 model.train()시에 forward 하도록, model.eval()시에는 하지 않도록 설정 가능.
"""
"""
class PowModel(torch.nn.Module):
    def __init__(self):
        super(PowModel, self).__init__()
        self.powParam = torch.nn.Parameter(torch.tensor(1.0))
        self.powBuff = torch.tensor(0.0) # 1번
        self.register_buffer('powBuff', torch.tensor(0.0)) # 2번

    def forward(self, x, q):
        self.powBuff = self.powBuff * 0.6 + x * 0.4#1번
        if self.training:#2번
            self.powBuff = self.powBuff * 0.6 + x * 0.4
        return self.powParam * self.powBuff ** q

class LogModel(torch.nn.Module):
    def __init__(self):
        super(LogModel, self).__init__()
        self.logParam = torch.nn.Parameter(torch.tensor(1.0))
        self.logBuff = torch.tensor(0.0) #1
        self.register_buffer('logBuff', torch.tensor(0.0)) #2

    def forward(self, x):
        self.logBuff = self.logBuff * 0.6 + x * 0.4 #1
        if self.training: #2
            self.logBuff = self.logBuff * 0.6 + x * 0.4
        return self.logParam * torch.log(self.logBuff)

class PowLogModel(torch.nn.Module):
    def __init__(self):
        super(PowLogModel, self).__init__()
        self.powModel = PowModel()
        self.logModel = LogModel()

    def forwardd(self, x, q):
        y = self.powModel(x,q)
        z = self.logModel(y)

powLogModel = PowLogModel()

for i in range(10):
    z = powLogModel(5.0, 3.0)
    print(i, 'z', z)
    print(i, 'powBuff', powLogModel.powModel.powBuff)
    print(i, 'logBuff', powLogModel.logModel.logBuff)

print('state', powLogModel.state_dict())
print('parameters', list(powLogModel.named_parameters()))
print('buffer', list(powLogModel.named_buffers()))#"""


# hook
"""
https://hongl.tistory.com/157
https://teamdable.github.io/techblog/PyTorch-Module
Module 사용시, 내부를 건드리지 않고 forward, backward 값들을 관찰하거나 수정할 때 사용.
- register_forward_pre_hook : forward 호출 전에 작동하도록 걸어둠, input 수정 가능.
- register_forward_hook : forward 후에 작동하도록 걸어둠, input 수정가능 하나 forward 영향 없음, output 수정 가능.
- register_backward_hook : backward 후에 작동하도록 걸어둠. grad_input, grad_output 확인가능.
  - gard 수정은 의미 파악이 어렵기 때문에 debug 용도로만 사용. 
"""
"""
class PowModel(torch.nn.Module):
    def __init__(self):
        super(PowModel, self).__init__()
        self.powParam = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, q):
        return self.powParam*x**q

class LogModel(torch.nn.Module):
    def __init__(self):
        super(LogModel, self).__init__()
        self.logParam = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.logParam * torch.log(x)

class PowLogModel(torch.nn.Module):
    def __init__(self):
        super(PowLogModel, self).__init__()
        self.powModel = PowModel()
        self.logModel = LogModel()

    def forward(self, x, q):
        y = self.powModel(x,q)
        z = self.logModel(y)
        return z

powLogModel = PowLogModel()

def forward_pre_hook(module, input):
    print('input', input)
    first_input, *reset_input = input
    print(first_input, *reset_input)
    return (first_input*2, *reset_input)

def forward_hook(module, input, output):
    print('input', input)
    print('output', output)
    return output * 2

def backward_hook(module, grad_input, grad_output):
  print('grad_input', grad_input)
  print('grad_output', grad_output)
  return grad_input

# forward 전, hook
#hook = powLogModel.logModel.register_forward_pre_hook(forward_pre_hook)
#hook = powLogModel.powModel.register_forward_pre_hook(forward_pre_hook)

# forward 후, hook
hook = powLogModel.logModel.register_forward_hook(forward_hook)

# backward 후, hook
hook = powLogModel.powModel.register_backward_hook(backward_hook)

x = torch.tensor(5.0, requires_grad=True)
q = torch.tensor(3.0, requires_grad=True)
z = powLogModel(x, q)
z.backward()

hook.remove()

print('x', x)
print('x.grad', x.grad)
print('q', q)
print('q.grad', q.grad)
print('z', z)

for name, parameter in powLogModel.named_parameters():
    print(name, f'data({parameter.data}), grad({parameter.grad}')#"""


#hook 사용한 feature extractor, gradient clipping
"""
#https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904

from torch import Tensor
from torchvision.models import resnet50


# forward hook 사용을 위한.
def module_hook(module: nn.Module, input: Tensor, output: Tensor):
    pass

# backward hook 사용을 위한.
def tensor_hook(grad: Tensor):
    pass#"""

# Ex.1) forward 후, hook register 각 layer name 할당.
"""
class VerboseExecusion(nn.Module):
    def __init__(self, model: nn.Module):
        super(VerboseExecusion,self).__init__()
        self.model = model

        # 각 layer hook 등록
        for name, layer in self.model.named_children():
            layer.__name__ = name   #forward가 끝나고 register 되므로 해줘야함. python 내장함수 __name__을 해줌으로써 attribute 추가.
            layer.register_forward_hook(lambda layer, _, output: print(f'{layer.__name__}: {output.shape}'))

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

verbose_reset = VerboseExecusion(resnet50())
dummy_input = torch.ones(10, 3, 224, 224)

_ = verbose_reset(dummy_input)#"""

# Ex.2) Feature Extraction.
"""
from typing import Dict, Iterable, Callable

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        # hook register
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    # hook 지정한 layer output 저장.
    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    # forward 돌리고, hook으로 저장한 layer 반환.
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features

resnet_features = FeatureExtractor(resnet50(), layers=["layer4", "avgpool"])
featurs = resnet_features(dummy_input)

print({name: output.shape for name, output in featurs.items()})#"""

#Ex.3) Gradient Cliping
"""
https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
그래디언트가 너무 크게 증가하는 것을 막는 방법. gradient clipping/normalization/modification
"""
"""
def gradient_clipper(model: nn.Module, val: float) -> nn.Module:
    for parameter in model.parameters():
        parameter.register_hook(lambda grad: grad.clamp_(-val, val)) #backward hook

    return model

clipped_resnet = gradient_clipper(resnet50(), 0.01)
pred = clipped_resnet(dummy_input)
loss = pred.log().mean()
loss.backward()

print(clipped_resnet.fc.bias.grad[:25])#"""





































