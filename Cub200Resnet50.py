from avalanche.benchmarks.classic import SplitCUB200
from torchvision import transforms
import os
import torch
import torchvision.models as models
import random
import numpy as np

featuresPath = './Features/cub200_resnet50'

bs = 50
seed = 317

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
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

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).cuda()
    model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.Sequential(*list(model.module.children())[:-1])

    return model

class CUB200RESNET50():
    def __init__(self, start = 100, step = 2):
        
        #self.alpha = 0.9
        #self.T = 2.3

        self.n_classes = 200
        self.n_features = 2048
        self.train_features, self.test_features = createFeatures(start, step)

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


def createFeatures(start = 100, step = 2):
    #print('Creating features....')
    set_seed()
    benchmark = SplitCUB200(n_experiences = ( (200 - start) // step) + 1, classes_first_batch=start, train_transform=train_transform, eval_transform=test_transform)
    
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
    print('features saved')

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
    ds = CUB200RESNET50(step = 2)
