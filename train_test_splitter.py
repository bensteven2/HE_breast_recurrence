# -*- coding: UTF-8 -*-

import os
import random
import shutil
from glob import glob
import pandas as pd
import argparse
# import yaml
from utils.common import logger
import os
from pathlib import Path
import sys
import numpy as np
from random import shuffle

from sklearn.model_selection import StratifiedKFold
##benben
classification = {
    '0': 0,
    '1': 1,
}

multiclass_list = ["健康对照组", "普通肺炎", "新冠肺炎"]


def sampling(X, y, K, t=0, types=2):
    if K == 1:
        TF_split_list = [(np.array( list(range(len(y))) ), np.array( list(range(len(y))) ))]
        print("TF_split_list:",TF_split_list)
    else:
        skf = StratifiedKFold(n_splits=K, shuffle=True)
        TF_split_list = skf.split(X, y)
    for fold, (train, test) in enumerate(TF_split_list):
        train_data = []
        test_data = []

        train_set, train_label = pd.Series(X).iloc[train].tolist(), pd.Series(y).iloc[train].tolist()
        test_set, test_label = pd.Series(X).iloc[test].tolist(), pd.Series(y).iloc[test].tolist()

        for (data, label) in zip(train_set, train_label):
            for img in glob(data+'/*'):
                train_data.append((img, label)) 
        for (data, label) in zip(test_set, test_label):
            for img in glob(data+'/*'):
                test_data.append((img, label))

        pdf = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(drop = True)
        
        # Get the smallest number of image in each category
        min_num = min(pdf['label'].value_counts())
        
        # Random downsampling
        index = []
        for i in range(2):
            if i == 0:
                start = 0
                end = pdf['label'].value_counts()[i]
            else:
                start = end
                end = end + pdf['label'].value_counts()[i]
            index = index + random.sample(range(start, end), min_num)
            
        pdf = pdf.iloc[index].reset_index(drop = True)

        # Shuffle
        pdf = pdf.reindex(np.random.permutation(pdf.index)).reset_index(drop = True)
        if types != 2:
            pdf.to_csv(os.getcwd()+f'/database_c{types}/train_ovr_{t}_fold_{fold}.csv', index=None, header=None)
        else:
            pdf.to_csv(os.getcwd()+f'/database_c{types}/train_{fold}.csv', index=None, header=None)

        pdf1 = pd.DataFrame(test_data)
        if types != 2:
            pdf1.to_csv(os.getcwd()+f'/database_c{types}/test_ovr_{t}_fold_{fold}.csv', index=None, header=None)
        else:
            pdf1.to_csv(os.getcwd()+f'/database_c{types}/test_{fold}.csv', index=None, header=None)


def split_set(path, types, K, shuffle=True):
    X  = []
    y = []
    img_path = glob(path+'/*')
    for ip in img_path:
        peoples = glob(ip+'/*')
        labels = [classification[Path(ip).stem]] * len(peoples)
        X.extend(peoples)
        y.extend(labels)  
    print("---y is:",y)
    sampling(X, y, K)


def split_set_for_multiclass(path, types, K, shuffle=True):
    total = glob(path+'/*/*')
    for t in range(types):
        one = glob(path+'/'+multiclass_list[t]+'/*')
        other = list(set(total).difference(set(one)))
        X = one + other
        y = [1] * len(one) + [0] * len(other)
        sampling(X, y, K, t, types)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--stained_tiles_home', type=str, default="/home/xisx/data/covid_19/")
    # parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--types', type=int, default=3)
    args = parser.parse_args()
    split_set(args.stained_tiles_home, args.types)  # , args.label_dir_path)
