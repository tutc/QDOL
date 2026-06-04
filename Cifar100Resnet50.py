from avalanche.benchmarks.classic import SplitCIFAR100
from torchvision import transforms
import os
import torch
import torchvision.models as models
import random
import numpy as np


featuresPath = './Features/cifar100_resnet50.pt'
pretrainedPath = './PretrainedModel/resnet50_ImageNet32.pth.tar'

bs = 50
seed = 317

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

def initFeaturesExtractor():

    assert os.path.isdir('PretrainedModel'), 'Error: no pretrained directory found!'
    assert os.path.isfile(pretrainedPath), 'Error: no pretrained file found!'
    checkpoint = torch.load(pretrainedPath)

    model = models.resnet50().cuda()
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.Sequential(*list(model.module.children())[:-1])
        
    return model

class CIFAR100RESNET50():
    def __init__(self, start = 2, step = 2):
        
        #self.alpha = 0.9
        #self.T = 2.3

        self.n_classes = 100
        self.n_features = 2048

        experiences = self.n_classes//step
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
    #print('Creating features....')
    set_seed()
    benchmark = SplitCIFAR100(n_experiences=experiences, train_transform=train_transform, eval_transform=test_transform)
    
    featuresExtrator = initFeaturesExtractor()

    train_features = []
    test_features = []


    for (train_exp, test_exp) in zip(benchmark.train_stream, benchmark.test_stream):
        current_train_set = train_exp.dataset
        current_test_set = test_exp.dataset
        
        current_train_features, current_test_features = getFeatures(featuresExtrator, current_train_set, current_test_set)

        #train_features.append(torch.utils.data.DataLoader(current_train_features, batch_size=bs, shuffle=True, num_workers=0))
        train_features.append(torch.utils.data.DataLoader(current_train_features, batch_size=bs, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed)))
        test_features.append(torch.utils.data.DataLoader(current_test_features, batch_size=bs, shuffle=False, num_workers=0))

    return train_features, test_features


def splitFeatures(trainset, testset, n_class, start, step):
    print('Spliting features.......')
    train_features=[]
    test_features=[]

    train_features.append(torch.utils.data.DataLoader(SubDataset(trainset,[j for j in range(start)]), batch_size=bs, shuffle=True, num_workers=0))
    test_features.append(torch.utils.data.DataLoader(SubDataset(testset,[j for j in range(start)]), batch_size=bs, shuffle=False, num_workers=0))
    
    for i in range(start, n_class, step):
        
        train_features.append(torch.utils.data.DataLoader(SubDataset(trainset,[i+j for j in range(step)]), batch_size=bs, shuffle=True, num_workers=0))
        test_features.append(torch.utils.data.DataLoader(SubDataset(testset,[i+j for j in range(step)]), batch_size=bs, shuffle=False, num_workers=0))

    return train_features, test_features



def getFeatures(model, trainset, testset):

    mini_bs = 1
    
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
    #for (data, target) in test_loader:
        data, target = data.cuda(), target.cuda()
        
        with torch.no_grad():
            output = model(data)
            output = output.view(output.size(0),-1)

            dict['testdata'] =  torch.cat((dict['testdata'], output))
            dict['testlabel'] =  torch.cat((dict['testlabel'], target))
    
    #saveFeatures(dict, featuresPath)

    train_set = torch.utils.data.TensorDataset(dict['traindata'], dict['trainlabel'])
    test_set = torch.utils.data.TensorDataset(dict['testdata'], dict['testlabel'])

    return train_set, test_set

def saveFeatures(dict, featuresPath = featuresPath):
    if os.path.isfile(featuresPath):
        old_dict = torch.load(featuresPath)
        old_dict['traindata'] = torch.cat((old_dict['traindata'], dict['traindata']))
        old_dict['trainlabel'] = torch.cat((old_dict['trainlabel'], dict['trainlabel']))
        old_dict['testdata'] = torch.cat((old_dict['testdata'], dict['testdata']))
        old_dict['testlabel'] = torch.cat((old_dict['testlabel'], dict['testlabel']))
        dict = old_dict
    torch.save(dict, featuresPath)

from torch.utils.data import Dataset
class SubDataset(Dataset):
    def __init__(self, original_dataset, sub_labels, target_transform=None,transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform
        self.transform=transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.transform:
            sample=self.transform(sample)
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


if __name__ == '__main__':
    initFeaturesExtractor()