import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 256)
        self.bn1_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2_fc = nn.BatchNorm1d(512)

    def forward(self, x,reverse=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 1152)#4096/1152
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        '''if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))'''
        return x


class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(1152, 128)#4096/1152
        self.bn1_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2_fc = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 2)############### CLASS
        self.bn_fc3 = nn.BatchNorm1d(2)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        # if reverse:
        #     x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x
