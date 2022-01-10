#CUDA_LAUNCH_BLOCKING=1
import time
import copy
import torch
from model import leaf_Net
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import data_dir_setting, data_loaders_leaf_classification


def train(model, train_loader, optimizer):
    train_loss = 0
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)

    return train_loss, train_accuracy


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


def train_baseline(model, train_loader, val_loader, test_loader, optimizer, num_epochs=30):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        since = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        test_loss, test_acc = evaluate(model, test_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('-------------- epoch {} ----------------'.format(epoch))
        print('train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))
        print('val Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_acc))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return model


BATCH_SIZE = 256
EPOCH = 30
data_setting = False
dataset_dir = 'D:\datasets\leaf\Plant_leaf_diseases_dataset'
base_dir = 'D:\datasets\leaf/splitted' # train/test/val


if __name__ == '__main__':
    if data_setting:
        data_dir_setting(dataset_dir, base_dir)

    train_loader, val_loader, test_loader = data_loaders_leaf_classification(base_dir, BATCH_SIZE)

    model_base = leaf_Net().cuda()

    optimizer = optim.Adam(model_base.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    base = train_baseline(model_base, train_loader, val_loader,test_loader, optimizer, EPOCH)  # (16)
    torch.save(base, 'pt/baseline.pt')
