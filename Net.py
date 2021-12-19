import torch
import torch.nn as nn
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling


class Cnn_With_Clinical_Net(nn.Module):
    def __init__(self , model):
        super(Cnn_With_Clinical_Net, self).__init__()
        # self.layer = nn.Sequential(*list(model.children())[:-1])
        # self.feature = list(model.children())[-1].in_features
        # self.cnn = nn.Linear(self.feature, 128)
        
        # CNN
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-1]
        self.dense = None
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1] 
        else:
            self.feature = self.layer[-1].in_features
        self.linear = nn.Linear(self.feature, 128)
        
        # clinical feature
        ###benben
        self.clinical = nn.Linear(5, 5)
        ###benben
        # concat
        self.mcb = CompactBilinearPooling(128, 5, 128).cuda()
        # self.concat = nn.Linear(128+55, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(True)
        self.classifier = nn.Linear(128, 2)

    def forward(self, x, clinical_features):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self.dense is not None:
            x = self.dense(x)
        x = self.linear(x)
        clinical = self.clinical(clinical_features)
        x = self.mcb(x, clinical)
        # x = torch.cat([x, clinical], dim=1)
        # x = self.concat(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

class LogisticRegression(nn.Module):
    def __init__(self, model):
        super(LogisticRegression, self).__init__()
        self.clinical = nn.Linear(5, 5)
        self.sm = nn.Sigmoid()
        self.classifier = nn.Linear(5, 2)
 
    def forward(self, x, clinical_features):
        x = self.clinical(clinical_features)
        x = self.sm(x)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-1]
        self.dense = None
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1] 
        else:
            self.feature = self.layer[-1].in_features
        self.linear = nn.Linear(self.feature, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self.dense is not None:
            x = self.dense(x)
        x = self.linear(x)
        return x
'''    
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.layer = nn.Sequential(*list(model.children()))
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.layer[-1][-1].out_features = 2
        else:
            self.layer[-1].out_features = 2
    
    def forward(self, x):
        x = self.layer(x)
        return x
'''


