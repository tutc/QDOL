import torch
import torch.nn as nn
from torchvision.models import resnet18

import Benchmarks as benchmarks

from avalanche.benchmarks.classic import SplitCIFAR10
from torchvision import transforms
import os
import torchvision.models as models
import random
import numpy as np

featuresPath = './Features/cifar10_resnet18.pt'
pretrainedPath = './PretrainedModel/resnet18_ImageNet32.pth.tar'

bs = 50
seed = 317


pretrained_path = './PretrainedModel/resnet18_ImageNet32.pth.tar'

def initFeaturesExtractor_new(device="cuda"):
    """
    CIFAR-style ResNet18 feature extractor
    - input: 32x32
    - output: 512-dim
    - pretrained on CIFAR (từ checkpoint cũ của bạn)
    """

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Identity()

    checkpoint = torch.load(pretrained_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k.startswith("fc."):
            continue            # 🔥 bỏ fc.weight, fc.bias
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)

    for p in model.parameters():
        p.requires_grad = False

    model.to(device)
    model.eval()
    return model


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def set_seed():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #print('Co vao set_seet() voi seed = ',seed)

def initFeaturesExtractor():
    assert os.path.isdir('PretrainedModel'), 'Error: no pretrained directory found!'
    assert os.path.isfile(pretrainedPath) , 'Error: no pretrained file found!'
    
    checkpoint = torch.load(pretrainedPath)


    model = models.resnet18().cuda()
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model = torch.nn.DataParallel(model).cuda()
        
    model.load_state_dict(checkpoint['state_dict'])    
    model = torch.nn.Sequential(*list(model.module.children())[:-1])
    

    return model


class CIFAR10RESNET18():
    def __init__(self, start = 2, step = 2):
        
        self.n_classes = 10
        self.n_features = 512
	
        experiences = self.n_classes // step
        self.train_features, self.test_features = createFeatures(experiences)

    def clone(dataset):
        """ Tạo bản sao dataset mà không khởi tạo lại object mới """
    
        # Sao chép DataLoader của train_features với generator giữ nguyên seed
        new_train_features = []
        for dl in dataset.train_features:
            gen_seed = dl.generator.initial_seed() if dl.generator is not None else None
            new_generator = torch.Generator()
            if gen_seed is not None:
                new_generator.manual_seed(gen_seed)

            new_dl = torch.utils.data.DataLoader(
                dl.dataset,
                batch_size=dl.batch_size,  # Giữ nguyên batch_size gốc
                shuffle=True,
                #sampler = dl.sampler,
                num_workers=dl.num_workers,
                generator=new_generator
            )
            new_train_features.append(new_dl)

        # Sao chép test_features (không cần shuffle, không cần generator)
        new_test_features = [
            torch.utils.data.DataLoader(
                dl.dataset,
                batch_size=dl.batch_size,
                shuffle=False,
                num_workers=dl.num_workers
            )
            for dl in dataset.test_features
        ]

        # Gán lại dataset để giữ nguyên object nhưng update thuộc tính
        dataset.train_features = new_train_features
        dataset.test_features = new_test_features

        return dataset  # Trả về object đã được cập nhật

def createFeatures(experiences):
    
    set_seed()

    print('Creating features....')
    benchmark = SplitCIFAR10(n_experiences=experiences, train_transform=train_transform, eval_transform=test_transform)

    featuresExtrator = initFeaturesExtractor()

    train_features = []
    test_features = []


    for (train_exp, test_exp) in zip(benchmark.train_stream, benchmark.test_stream):
        current_train_set = train_exp.dataset
        current_test_set = test_exp.dataset
        
        current_train_features, current_test_features = getFeatures(featuresExtrator, current_train_set, current_test_set)
        train_features.append(torch.utils.data.DataLoader(current_train_features, batch_size=bs, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed)))
        test_features.append(torch.utils.data.DataLoader(current_test_features, batch_size=bs, shuffle=False, num_workers=0))

    return train_features, test_features

'''
def getFeatures(model, trainset, testset):

    mini_bs = 256

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=mini_bs, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=mini_bs, num_workers=0)


    empty = torch.tensor([]).cuda()
    dict = {'traindata': empty, 'trainlabel':empty, 'testdata': empty, 'testlabel':empty}

    model.eval()
    for (data, target, _) in train_loader:             #Avalanche
    #for (data, target) in train_loader:

        data, target = data.cuda(), target.cuda()
        
        with torch.no_grad():
            output = model(data)
            output = output.view(output.size(0),-1)

            dict['traindata'] =  torch.cat((dict['traindata'], output))
            dict['trainlabel'] =  torch.cat((dict['trainlabel'], target))

    for (data, target, _) in test_loader:      #Avalanche
        data, target = data.cuda(), target.cuda()
        
        with torch.no_grad():
            output = model(data)
            output = output.view(output.size(0),-1)

            dict['testdata'] =  torch.cat((dict['testdata'], output))
            dict['testlabel'] =  torch.cat((dict['testlabel'], target))
    
    #torch.save(dict, featuresPath)

    train_set = torch.utils.data.TensorDataset(dict['traindata'], dict['trainlabel'])
    test_set = torch.utils.data.TensorDataset(dict['testdata'], dict['testlabel'])

    return train_set, test_set
'''

def getFeatures(model, trainset, testset):

    mini_bs = 256

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=mini_bs, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=mini_bs, shuffle=False, num_workers=0
    )

    model.eval()

    # ===== TRAIN =====
    train_feats = []
    train_labels = []
    train_indices = []

    with torch.no_grad():
        for data, target, idx in train_loader:  # Avalanche trả idx
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(data)
            output = output.view(output.size(0), -1)

            train_feats.append(output.cpu())
            train_labels.append(target.cpu())
            train_indices.append(idx.cpu())

    # concat
    train_feats = torch.cat(train_feats, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    train_indices = torch.cat(train_indices, dim=0)

    # sort theo index gốc (QUAN TRỌNG)
    order = torch.argsort(train_indices)
    train_feats = train_feats[order]
    train_labels = train_labels[order]

    train_set = torch.utils.data.TensorDataset(train_feats, train_labels)

    # ===== TEST =====
    test_feats = []
    test_labels = []
    test_indices = []

    with torch.no_grad():
        for data, target, idx in test_loader:
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(data)
            output = output.view(output.size(0), -1)

            test_feats.append(output.cpu())
            test_labels.append(target.cpu())
            test_indices.append(idx.cpu())

    test_feats = torch.cat(test_feats, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    test_indices = torch.cat(test_indices, dim=0)

    order = torch.argsort(test_indices)
    test_feats = test_feats[order]
    test_labels = test_labels[order]

    test_set = torch.utils.data.TensorDataset(test_feats, test_labels)

    return train_set, test_set


if __name__ == '__main__':
    dataset = benchmarks.Cifar10Resnet18_New.CIFAR10RESNET18()
    
    x, y = next(iter(dataset.train_features[0]))
    print(x.shape)   # phải là [B, 512]
    print(x.std())   # KHÔNG được ~0

