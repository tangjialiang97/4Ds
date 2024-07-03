import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.utils.data as data
import torch
seed = 42
torch.manual_seed(seed)
from PIL import Image
import os
import random
import numpy as np
from torch.utils.data import Dataset

class ImageList(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0],
                       np.array([int(la) for la in val.split()[1:]]))
                      for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1]))
                      for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList_idx(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

def get_pacs_target_loader(data_path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = datasets.ImageFolder(data_path,img_transform)
    dataset.num_classes = 7

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def get_source_target_loader(dataset_name, source_path, target_path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):

    source_dataloader = None
    if dataset_name == "Office31":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_test_loader(target_path, batch_size, num_workers, pin_memory)

    elif dataset_name == "ImageClef":
        if source_path != "":
            source_dataloader = imageclef_train_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = imageclef_train_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = imageclef_test_loader(target_path, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")

    return source_dataloader, target_dataloader, target_testloader

def get_m_source_target_loader(dataset_name, source_path_1, source_path_2, target_path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):

    source_dataloader_1 = None
    source_dataloader_2 = None
    if dataset_name == "Office31":
        if source_path_1 != "":
            source_dataloader_1 = office_loader(source_path_1, batch_size, num_workers, pin_memory, drop_last)
        if source_path_2 != "":
            source_dataloader_2 = office_loader(source_path_2, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_test_loader(target_path, batch_size, num_workers, pin_memory)

    elif dataset_name == "ImageClef":
        if source_path_1 != "":
            source_dataloader_1 = imageclef_train_loader(source_path_1, batch_size, num_workers, pin_memory, drop_last)
        if source_path_2 != "":
            source_dataloader_2 = imageclef_train_loader(source_path_2, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = imageclef_train_loader(target_path, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = imageclef_test_loader(target_path, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")

    return source_dataloader_1, source_dataloader_2, target_dataloader, target_testloader

def get_source_concat_target_loader(dataset_name, source_path, target_paths, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):

    source_dataloader = None
    if dataset_name == "Office31":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_m_concat_loader(target_paths, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_m_concat_test_loader(target_paths, batch_size, num_workers, pin_memory)
    elif dataset_name == "ImageClef":
        if source_path != "":
            source_dataloader = imageclef_train_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        raise ("To be implemented")
    elif dataset_name == "OfficeHome":
        if source_path != "":
            source_dataloader = office_loader(source_path, batch_size, num_workers, pin_memory, drop_last)
        target_dataloader = office_m_concat_loader(target_paths, batch_size, num_workers, pin_memory, drop_last)
        target_testloader = office_m_concat_test_loader(target_paths, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")


    return source_dataloader, target_dataloader, target_testloader

def get_train_test_loader(dataset_name, data_path, batch_size=16, num_workers=1, pin_memory=False):
    if dataset_name.lower() == "office31":
        train_loader, test_loader = office_train_test_loader(data_path, batch_size, num_workers, pin_memory)
    elif dataset_name.lower() == "officehome":
        train_loader, test_loader = office_train_test_loader(data_path, batch_size, num_workers, pin_memory)
    elif dataset_name.lower() == "officecaltech":
        train_loader, test_loader = office_train_test_loader(data_path, batch_size, num_workers, pin_memory)
    elif dataset_name.lower() == "pacs":
        train_loader, test_loader = office_train_test_loader(data_path, batch_size, num_workers, pin_memory)
    elif dataset_name.lower() == "domainnet":
        train_loader, test_loader = office_train_test_loader(data_path, batch_size, num_workers, pin_memory, split_ratio=0.9)
    elif dataset_name.lower() == "visda":
        train_loader, test_loader = imageclef_train_test_loader(data_path, batch_size, num_workers, pin_memory)

    elif dataset_name == "ImageClef":
        train_loader, test_loader = imageclef_train_test_loader(data_path, batch_size, num_workers, pin_memory)
    else:
        raise("Dataset not handled")

    return train_loader, test_loader

def get_train_test_loader_aug(dataset_name, data_path, batch_size=16, num_workers=1, pin_memory=False):
    if dataset_name.lower() == "officehome":
        train_loader, train_loader_aug, test_loader = office_train_test_loader_aug(data_path, batch_size, num_workers, pin_memory)
    elif dataset_name.lower() == "officecaltech":
        train_loader, train_loader_aug, test_loader = office_train_test_loader_aug(data_path, batch_size, num_workers, pin_memory)
    elif dataset_name.lower() == "pacs":
        train_loader, train_loader_aug, test_loader = office_train_test_loader_aug(data_path, batch_size, num_workers, pin_memory)
    elif dataset_name.lower() == "domainnet":
        train_loader, train_loader_aug, test_loader = office_train_test_loader_aug(data_path, batch_size, num_workers, pin_memory, split_ratio=0.9)



    return train_loader, train_loader_aug, test_loader

def office_train_test_loader(path, batch_size=16, num_workers=1, pin_memory=False, split_ratio=0.8):
    image_list = open(os.path.join(path, "image_list.txt")).readlines()
    random.seed(42)  # 使用任意整数作为种子
    random.shuffle(image_list)
    split_point = int(split_ratio * len(image_list))

    # 分割为训练集和测试集
    list_train = image_list[:split_point]
    list_test = image_list[split_point:]

    transforms_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = ImageList(list_train,
                                 transform=transforms_train)
    test_set = ImageList(list_test,
                                transform=transforms_test)

    train_loader = data.DataLoader(train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    test_loader = data.DataLoader(test_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    return train_loader, test_loader

def office_train_test_loader_aug(path, batch_size=16, num_workers=1, pin_memory=False, split_ratio=0.8):
    image_list = open(os.path.join(path, "image_list.txt")).readlines()
    random.seed(42)  # 使用任意整数作为种子
    random.shuffle(image_list)
    split_point = int(split_ratio * len(image_list))

    # 分割为训练集和测试集
    list_train = image_list[:split_point]
    list_test = image_list[split_point:]

    transforms_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformations_aug = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms.RandomGrayscale(p=0.2),
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = ImageList(list_train,
                                 transform=transforms_train)

    train_set_aug = ImageList(list_train,
                              transform=transformations_aug)

    test_set = ImageList(list_test,
                                transform=transforms_test)

    train_loader = data.DataLoader(train_set,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=pin_memory)
    train_loader_aug = data.DataLoader(train_set_aug,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory)

    test_loader = data.DataLoader(test_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    return train_loader, train_loader_aug, test_loader

def office_loader(path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    dataset.num_classes = len(dataset.classes)

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def office_m_concat_loader(paths, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    datasetsF = []
    for p in paths:
        datasetsF.append(datasets.ImageFolder(p,
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize,
                                       ])))
    dataset = torch.utils.data.ConcatDataset(datasetsF)
    dataset.num_classes = len(datasetsF[0].classes)

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def office_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = datasets.ImageFolder(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

def office_m_concat_test_loader(target_paths, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    datasetsF = []

    for p in target_paths:
        datasetsF.append(datasets.ImageFolder(p,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize,
                                   ])))

    dataset = torch.utils.data.ConcatDataset(datasetsF)
    dataset.num_classes = len(datasetsF[0].classes)

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

def imageclef_train_loader(path, batch_size=16, num_workers=1, pin_memory=False, drop_last=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = CLEFImage(path, transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]))

    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=drop_last)

def imageclef_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = CLEFImage(path, transforms.Compose([
                           transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           normalize,
                       ]))

    return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

def visda_train_test_loader(type, path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'validation')
    train_set = CLEFImage_visda(train_path,
                               transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
    val_set = CLEFImage_visda(val_path,
                          transforms.Compose([
                              transforms.Resize(256),
                              transforms.RandomResizedCrop(224),
                              transforms.ToTensor(),
                              normalize,
                          ]))

    train_loader = data.DataLoader(train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    test_loader = data.DataLoader(val_set,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    return train_loader, test_loader


def imageclef_train_test_loader(path, batch_size=16, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = CLEFImage(path,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))

    train_set, test_set = data.random_split(dataset,
                                            [int(0.7 * dataset.__len__()), dataset.__len__() - int(0.7 * dataset.__len__())])

    train_loader = data.DataLoader(train_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    test_loader = data.DataLoader(test_set,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory)

    return train_loader, test_loader

def default_loader(path):
    return Image.open(path).convert('RGB')

def chekc_data(train_loader, test_loader):
    print('The number of train data is {}'.format(len(train_loader.dataset)))
    print('The number of test data is {}'.format(len(test_loader.dataset)))
    for i, (input, target) in enumerate(train_loader):
        print(input.shape)
        print(target.shape)
        break
    for i, (input, target) in enumerate(test_loader):
        print(input.shape)
        print(target.shape)
        break
def make_imageclef_dataset(path):
    images = []
    dataset_name = path.split("/")[-1]


    label_path = os.path.join(path, ".." , "list", "{}List.txt".format(dataset_name))
    image_folder = os.path.join(path, "..", "{}".format(dataset_name))
    labeltxt = open(label_path)

    for line in labeltxt:
        pre_path, label = line.strip().split(' ')
        image_name = pre_path.split("/")[-1]
        image_path = os.path.join(image_folder, image_name)

        gt = int(label)
        item = (image_path, gt)
        images.append(item)
    return images

def make_imageclef_visda(path, dataset=None):
    images = []


    label_path = os.path.join(path, "image_list.txt")
    image_folder = path

    labeltxt = open(label_path)

    for line in labeltxt:
        pre_path, label = line.strip().split(' ')
        # image_name = pre_path.split("/")[-1]
        image_name = pre_path
        image_path = os.path.join(image_folder, image_name)

        gt = int(label)
        item = (image_path, gt)
        images.append(item)
    return images

class CLEFImage(data.Dataset):
    def __init__(self, root, transform=None, image_loader=default_loader, dataset=None):
        imgs = make_imageclef_dataset(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.image_loader = image_loader
        self.num_classes = 12

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.image_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

class CLEFImage_visda(data.Dataset):
    def __init__(self, root, transform=None, image_loader=default_loader):
        imgs = make_imageclef_visda(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.image_loader = image_loader
        self.num_classes = 12

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.image_loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)