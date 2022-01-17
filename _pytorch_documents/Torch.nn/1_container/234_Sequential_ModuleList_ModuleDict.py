import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Sequential
"""
nn.Sequential 안에 layer을 쌓아 두면 각 layer의 forward가 연결됨.
입력값이 들어오면 각 layer forward를 순차적으로 수행하고 결과를 반환함. 
OrderDict과 같은 형태이고, nn.Sequential은 generator 구조.
"""
"""
model = nn.Sequential(
    nn.Conv2d(1,20,5),
    nn.ReLU(),
    nn.Conv2d(20,64,5),
    nn.ReLU()
)

model2 = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1,20,5)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(20,64,5)),
    ('relu2', nn.ReLU())
]))

# 예시
def make_sequential(in_channels, out_channels, *args, **kwargs):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, *args, **kwargs),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(),
                         nn.MaxPool2d(*args, **kwargs))


class CNNModel(nn.Module):
    def __init__(self, class_num, hidden_dim=32, out_dim=64):
        super().__init__()
        self.layer1 = make_sequential(1,hidden_dim, kernel_size=5, stride=1, padding=2)
        self.layer2 = make_sequential(hidden_dim, out_dim, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(out_dim,class_num)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

cnn_model = CNNModel(10,32,64)
input = torch.randn(1,1,100,100)
out = cnn_model(input)#"""


#ModuleList
"""
모듈들을 리스트로 묶음. 
여러 개의 동일한 구조의 다른 layer들을 리스트로 묶을 수 있다.
optimizer시에 토치 프레임워크가 해당 모듈 리스트를 인지함. 
- append, extend, insert 기능이 있음. extend는 iterable 한 것을 append 함. 
"""
"""
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,10) for _ in range(10)])
        self.linears.extend([nn.Linear(10,10) for _ in range(10)])

    def forward(self, x):
        # ModuleList는 iterable, indexing 할 수 있음.
        for i, l in enumerate(self.linears):
            x = self.linears[i//2](x) + l(x)
        return x
model = MyModule()
input = torch.randn(10,10)
out = model(input)
#"""

#nn.Sequential vs nn.ModuleList
"""
https://dongsarchive.tistory.com/67?category=329289
- nn.Sequential : 안에 있는 모듈 연결, 하나의 뉴럴넷.
- nn.ModuleList : 개별적으로 모듈들 담겨 있는 리스트.
  - python list와 다른 점은 토치 프레임워크가 모듈이라고 인식하고 optimizer 가능. 

언제 적절히 사용할 것인가. 예문.
https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
"""
"""
# - nn.ModuleList
class LinearNet(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size):
        super(LinearNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
        self.linears.extend([nn.Linear(layers_size, layers_size) for _ in range(1,num_layers-1)])
        self.linears.append(nn.Linear(layers_size, output_size))

# - nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return  x.view(N, -1)

simple_cnn = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=7, stride=2),
    nn.ReLU(inplace=True),
    Flatten(),
    nn.Linear(5408, 10)
)#"""


#ModuelDict
"""
모듈을 딕셔너리 형태로 모은다. 기존 딕셔너리 처럼 인덱싱 가능.
- clear() : 모든 items 제거. 안에 저장한 것들 다 제거.
- items() : return key/value
- keys(), pop(key), values()
- update(modules) 모르게씅.
"""
"""
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
            'conv':nn.Conv2d(1,10,3),
            'pool':nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()]
        ])
        self.linear = nn.Linear(10,1)

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        x = F.avg_pool2d(x, x.size()[3]).view(x.shape[0],-1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

model = MyModule()
input = torch.randn(1,1,100,100)
out = model(input, 'conv', 'lrelu')
print(out)#"""















