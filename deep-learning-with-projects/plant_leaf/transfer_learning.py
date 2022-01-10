import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import torch
import time
from torchvision import models
from torch.optim import lr_scheduler
from utils import data_loaders_transfer_learning


def train_resnet(model, criterion, optimizer, scheduler,dataloaders, dataset_sizes, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('-------------- epoch {} ----------------'.format(epoch + 1))
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    return model


def transfer_learning(dataloader, datasize):
    resnet = models.resnet50(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 39)
    resnet = resnet.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.001)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # freeze. ~layer1까지 고정.
    ct = 0
    for child in resnet.children(): #자식 모듈을 반복 가능한 객체로 변환.
        ct += 1
        if ct < 6:
            for param in child.parameters():
                param.requires_grad = False
    # 가장 best 모델 받아옴.
    model_resnet50 = train_resnet(resnet, criterion, optimizer_ft, exp_lr_scheduler,dataloader, datasize, num_epochs=EPOCH)

    torch.save(model_resnet50, save_root + '/resnet50.pt')


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, test_accuracy


BATCH_SIZE = 256
EPOCH = 30
data_setting = False
dataset_dir = 'D:\datasets\leaf\Plant_leaf_diseases_dataset'
base_dir = 'D:\datasets\leaf/splitted' # train/test/val
transfer = True # True=학습, False=평가
data_root = 'D:\datasets\leaf/splitted'
save_root = 'pt'


if __name__ == '__main__':
    dataloaders, dataset_sizes, test_loader_resNet = data_loaders_transfer_learning(data_root, BATCH_SIZE, transfer)
    if transfer:
        transfer_learning(dataloaders, dataset_sizes)

    resnet50 = torch.load(save_root + '/resnet50.pt')
    resnet50.eval()
    test_loss, test_accuracy = evaluate(resnet50, test_loader_resNet)

    print('ResNet test acc:  ', test_accuracy)


