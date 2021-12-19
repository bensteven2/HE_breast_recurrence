
import argparse
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
import numpy as np
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    #parser.add_argument('--classification', type=int, default=2)
    #parser.add_argument('--image_root', type=str, default="/data2/ben/data/TCGA/breast_torch")
    parser.add_argument('--clinical_json_address', type=str, default="./clinical.json")
    parser.add_argument('--input_csv_file_address', type=str, default="")
    parser.add_argument('--input_type', type=str, default="train")
    parser.add_argument('--label_name', type=str, default="patient,type,age,tumor_stage")
    #parser.add_argument('--K', type=int, default=5)
    #parser.add_argument('--num_epochs', type=int, default=50)
    #parser.add_argument('--clinical', type=bool, default=False)
    #parser.add_argument('--only_test', type=bool, default=False)
    args = parser.parse_args()


    label_name_split = args.label_name.split(",")
    if(len(label_name_split) == 2):
        label_name_split.append('')

    if(len(label_name_split) <= 1):
        exit(1)


    data=pd.read_csv(args.input_csv_file_address, usecols=label_name_split )
    data = data.loc[data["type"]==args.input_type]
    data_list = data.values.tolist()

    data_ID = np.array(data_list)[:,1].tolist()
    data_data = np.array(data_list)[:,2:].tolist() 
    data_dict = {k: v for k,v in zip(data_ID,data_data)}

    with open(args.clinical_json_address,'w',encoding='utf-8') as  json_file:
        json.dump(data_dict,fp=json_file,ensure_ascii=False)



