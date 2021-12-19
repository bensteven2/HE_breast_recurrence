from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
import time
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.common import logger
from utils.ImageFolderPaths import ImageFolderWithPaths
from utils.custom_dset import CustomDset
from utils.analytics import draw_roc, draw_roc_for_multiclass

from train_test_splitter import split_set, split_set_for_multiclass
from train import train_model
from test import test

from Net import Net, Cnn_With_Clinical_Net, LogisticRegression

plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        #transforms.Resize(224),
        transforms.RandomCrop((224, 224)),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.FiveCrop(224),
        #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        #transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generative_model(model, classification, k, c=0, clinical=False, HE_exist=True, clinical_json_address="./clinical.json", num_epochs=50):
    if classification == 2:
        image_datasets = {x: CustomDset(os.getcwd()+f'/database_c{classification}/{x}_{k}.csv',  ##benben
                            data_transforms[x]) for x in ['train']}
    else:
        image_datasets = {x: CustomDset(os.getcwd()+f'/database_c{classification}/{x}_ovr_{c}_fold_{k}.csv',
                            data_transforms[x]) for x in ['train']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                    shuffle=True, num_workers=4) for x in ['train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    class_names = image_datasets['train'].classes

    logger.info(f'model {model} / 第 {k+1} 折')

    if model == "resnet18":
        model_ft = models.resnet18(pretrained=True)
    if model == "resnet50":
        model_ft = models.resnet50(pretrained=True)
    if model == "vgg19":
        model_ft = models.vgg19(pretrained=True)
    if model == "vgg16":
        model_ft = models.vgg16(pretrained=True)
    if model == "alexnet":
        model_ft = models.alexnet(pretrained=True)
    if model == "inception":
        model_ft = models.inception_v3(pretrained=True)
    
    if not HE_exist:
        model_ft = LogisticRegression(model_ft)
    elif clinical:
        model_ft = Cnn_With_Clinical_Net(model_ft)
    
    else:
        model_ft = Net(model_ft)

    print("model_ft:", model_ft)
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft, tb = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, 
        dataset_sizes, num_epochs=num_epochs, clinical=clinical, clinical_json_address=clinical_json_address )  ##benben
    tb.close()

    save_model = os.getcwd()+f'/results/models/{model}_c{classification}_{k}'
    if classification != 2:
        save_model = save_model + f'_ovr_{c}'
    if clinical:
        save_model = save_model + '_clinical'
    save_model = save_model + '.pkl'
    
    torch.save(model_ft, save_model)


def main(model_type, ocs, classification, K, clinical, HE_exist, only_test, clinical_json_address, num_epochs, test_data_file="/database_c2/test.csv", merge=False,test_from_my_file=False):
    if classification != 2:
        # Divide the data set only once
        # split_set_for_multiclass(ocs[classification], classification, K)
        
        # create and train model
        # for model in model_type:
        #     for c in range(classification):
        #         for k in range(K):
        #             generative_model(model, classification, k, c=c)

        # Evaluate model performance by roc
        # generative_model_for_multiclass(model_type, classification, K)
        draw_roc_for_multiclass(model_type, classification, K)
    else:
        if only_test == False:
            split_set(ocs[classification], classification, K)

            for model in model_type:
                for k in range(K):
                    generative_model(model, classification, k, clinical=clinical, HE_exist=HE_exist,  clinical_json_address=clinical_json_address, num_epochs=num_epochs)
        draw_roc(model_type, classification, K, merge=merge,clinical=clinical, test_from_my_file=test_from_my_file, clinical_json_address=clinical_json_address, test_data_file=test_data_file)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--classification', type=int, default=2)
    parser.add_argument('--image_root', type=str, default="/data2/ben/data/TCGA/breast_torch")
    parser.add_argument('--clinical_json_address', type=str, default="./clinical.json")
    parser.add_argument('--test_data_file', type=str, default="/database_c2/test.csv")
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--clinical', type=str2bool, default=False)
    parser.add_argument('--HE_exist', type=str2bool, default=False)
    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--roc_merge', type=str2bool, default=True)
    parser.add_argument('--test_from_my_file', type=str2bool, default=False)
    args = parser.parse_args()

    model_type = ("resnet50",) # "vgg16", "vgg19", "resnet18", "alexnet")
    origin_classfication_set = {
        2 : args.image_root,
        #2 : '/data2/ben/data/zhongzhong/breast_torch',
        3 : '/home/xisx/data/CT', 
    }

    main(model_type=model_type, ocs=origin_classfication_set, classification=args.classification, K=args.K, clinical=args.clinical, HE_exist=args.HE_exist,  only_test=args.only_test, clinical_json_address=args.clinical_json_address, num_epochs=args.num_epochs, test_data_file=args.test_data_file, merge=args.roc_merge,test_from_my_file=args.test_from_my_file)

