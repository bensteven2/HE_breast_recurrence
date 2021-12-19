import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F

import os
from utils.custom_dset import CustomDset
from utils.common import logger
import json
import sys
from sklearn import preprocessing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def test(model, classification, k=0, K=10, types=0, clinical=False, test_from_my_file=False, clinical_json_address="./clinical.json", test_data_file="/database_c2/test.csv"):
    model.eval()

    if clinical:
        with open(clinical_json_address, encoding='utf-8') as f:
            json_dict = json.load(f)
        peoples = [i for i in json_dict]
        features = np.array([json_dict[i] for i in json_dict], dtype=np.float32)
        min_max_scaler = preprocessing.MinMaxScaler()
        clinical_features = min_max_scaler.fit_transform(features)
    
    if test_from_my_file:
        print("in test_from_my_file == True")
        testset = CustomDset(test_data_file, data_transforms['test'])

    elif classification != 2:
        testset = CustomDset(os.getcwd()+f'/database_c{classification}/test_ovr_{types}_fold_{k}.csv', data_transforms['test'])
    else:
        testset = CustomDset(os.getcwd()+f'/database_c2/test_{k}.csv', data_transforms['test'])
    print("==================================test_from_my_file:",test_from_my_file)
    print("test_data_file:",test_data_file)
    print("os.getcwd:",os.getcwd())

    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=4)

    person_prob_dict = dict()
    ##
    all_result_address = test_data_file + "_" + str(k) + ".all_result.csv"
    all_result_person_address = test_data_file + "_" + str(k) + ".all_result_person.csv"
    if os.path.exists(all_result_address):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(all_result_address)
        #os.unlink(path)
    if os.path.exists(all_result_person_address):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(all_result_person_address)
        #os.unlink(path)
    file_handle=open(all_result_address,mode='a')
    file_person_handle=open(all_result_person_address,mode='a')
    item_count = 0
    all_result_item = " \n"
    all_result_item_person= ""
    with torch.no_grad():
        for data in testloader:
            images, labels, names, images_names = data
            if clinical:
                X_train_minmax = [clinical_features[peoples.index(i)] for i in names]
                outputs = model(images.to(device), torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))
            else:
                outputs = model(images.to(device))
            probability = F.softmax(outputs, dim=1).data.squeeze()
            probs = probability.cpu().numpy()
            #print("++++++++++++++++++++++++++probs:",probs)
            #print("end")
            for i in range(labels.size(0)):
                p = names[i]
                if p not in person_prob_dict.keys():
                    person_prob_dict[p] = {
                        'prob_0': 0, 
                        'prob_1': 0,
                        'label': labels[i].item(),
                        'img_num': 0}
                if probs.ndim == 2:
                    person_prob_dict[p]['prob_0'] += probs[i, 0]
                    person_prob_dict[p]['prob_1'] += probs[i, 1]
                    person_prob_dict[p]['img_num'] += 1
                else:
                    person_prob_dict[p]['prob_0'] += probs[0]
                    person_prob_dict[p]['prob_1'] += probs[1]
                    person_prob_dict[p]['img_num'] += 1
                try:
                    all_result_item = names[i] +  "," + images_names[i] + "," + str(labels[i].item()) + ","+ str(probs[i,1])
                except:
                    print("In cecept")
                    pass
                item_count = item_count + 1
                #all_result_item = names[i] + " " +  str(labels[i].item()) + " "+ str(probs[i,1])
                #print(names[i].item(), " ",images_names[i].item(), " ",labels[i].item(), " ", probs[i,1])
                #print(str(item_count)," ====item:",all_result_item)
                file_handle.write(all_result_item + '\n')
    #print("++++++++++++++++++++++++++person_prob_dict:",person_prob_dict)
    file_handle.close()
    label_list = []
    score_list = []
    for key in person_prob_dict:
        score = [
            person_prob_dict[key]['prob_0']/person_prob_dict[key]['img_num'],
            person_prob_dict[key]['prob_1']/person_prob_dict[key]['img_num'],
        ]
        score_list.append(score)
        label_list.append(person_prob_dict[key]['label'])
        all_result_item_person = str(key) + "," + str(person_prob_dict[key]['prob_1']/person_prob_dict[key]['img_num']) + "," + str(person_prob_dict[key]['label'])
        file_person_handle.write(all_result_item_person + '\n')
    file_person_handle.close()
    logger.info("0 class people number is {}, 1 class people number is {}, proportion is {}".format(
        label_list.count(0), label_list.count(1), label_list.count(0)/label_list.count(1)))

    total = len(person_prob_dict)
    correct = 0
    for key in person_prob_dict.keys():
        predict = 0
        if person_prob_dict[key]['prob_0'] < person_prob_dict[key]['prob_1']:
            predict = 1
        if person_prob_dict[key]['label'] == predict:
            correct += 1
    logger.info('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))

    return label_list, score_list
