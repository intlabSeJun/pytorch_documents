import torch.nn as nn
import torch.nn.functional as F

class leaf_Net(nn.Module):
    def __init__(self):
        super(leaf_Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1) #채널 수, 출력 채널 수, 커널 크기
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 39)

    def forward(self, x): #(N,3,256,256)
        x = self.conv1(x) #(N,32,64,64)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x