import torch


# parameter
"""
what is differenct 'Parameter', 'Linear()', 'Embedding'
https://audreywongkg.medium.com/pytorch-nn-parameter-vs-nn-linear-2131e319e463
● 결론
 - Parameter : 매우 작은 값 생성.
 - Linear : kaiming_uniform 사용.(=He 초기화)
 - Embedding : Linear과 동일.
"""
"""
weight = torch.nn.Parameter(torch.FloatTensor(2,2))
print(f'Parameter : {weight}')
weight = torch.nn.Linear(2,2,bias=False)
print(f'Linear : {weight.weight}')
weight = torch.nn.Embedding(2,2)
print(f'Embeddgin : {weight.weight}')#"""

#UninitializedParameter
# -> 모르겠음.




