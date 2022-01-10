import os
import shutil
import math
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def data_dir_setting(dataset_dir, base_dir):
    original_dataset_dir = dataset_dir
    classes_list = os.listdir(original_dataset_dir)

    base_dir = base_dir
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'val')
    if not os.path.isdir(validation_dir):
        os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    for cls in classes_list:
        if not os.path.isdir(os.path.join(train_dir, cls)):
            os.mkdir(os.path.join(train_dir, cls))
        if not os.path.isdir(os.path.join(validation_dir, cls)):
            os.mkdir(os.path.join(validation_dir, cls))
        if not os.path.isdir(os.path.join(test_dir, cls)):
            os.mkdir(os.path.join(test_dir, cls))


    for cls in classes_list:
        path = os.path.join(original_dataset_dir, cls)
        fnames = os.listdir(path)

        train_size = math.floor(len(fnames) * 0.6)
        validation_size = math.floor(len(fnames) * 0.2)
        test_size = math.floor(len(fnames) * 0.2)

        train_fnames = fnames[:train_size]
        print("Train size(", cls, "): ", len(train_fnames))
        for fname in train_fnames:
            src = os.path.join(path, fname)
            dst = os.path.join(os.path.join(train_dir, cls), fname)
            shutil.copyfile(src, dst)

        validation_fnames = fnames[train_size:(validation_size + train_size)]
        print("Validation size(", cls, "): ", len(validation_fnames))
        for fname in validation_fnames:
            src = os.path.join(path, fname)
            dst = os.path.join(os.path.join(validation_dir, cls), fname)
            shutil.copyfile(src, dst)

        test_fnames = fnames[(train_size + validation_size):(validation_size + train_size + test_size)]

        print("Test size(", cls, "): ", len(test_fnames))
        for fname in test_fnames:
            src = os.path.join(path, fname)
            dst = os.path.join(os.path.join(test_dir, cls), fname)
            shutil.copyfile(src, dst)


def data_loaders_leaf_classification(root, BATCH_SIZE):
    transform_base = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    """
    ImageFolder 매서드
     - 하나의 클래스가 하나의 폴더에 있는 구조일 때 사용. 
     - root 옵션으로 데이터 불러올 경로.
     
    Dataloader 
     - 불러온 이미지 데이터를 주어진 조건에 따라 미니 배치 단위로 분리하는 역할.
    """
    train_dataset = ImageFolder(root=root + '/train', transform=transform_base)
    val_dataset = ImageFolder(root=root + '/val', transform=transform_base)
    test_dataset = ImageFolder(root=root + '/test', transform=transform_base)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    return train_loader, val_loader, test_loader


def data_loaders_transfer_learning(root, BATCH_SIZE, transfer):
    dataloaders = None
    dataset_sizes = None
    if transfer:
        data_transforms = {
            'train': transforms.Compose([transforms.Resize([64, 64]),
                                         transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                         transforms.RandomCrop(52), transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

            'val': transforms.Compose([transforms.Resize([64, 64]),
                                       transforms.RandomCrop(52), transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        }

        data_dir = root
        image_datasets = {x: ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x]) for x in
                          ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        class_names = image_datasets['train'].classes

    transform_resNet = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_resNet = ImageFolder(root=root + '/test', transform=transform_resNet)
    test_loader_resNet = torch.utils.data.DataLoader(test_resNet, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    return dataloaders, dataset_sizes, test_loader_resNet

