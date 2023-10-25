import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, pretrain=False):
        super(Classifier, self).__init__()
        self.pretrain = pretrain
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(self.bottleneck,self.fc,nn.Softmax(dim=-1))
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(self.fc,nn.Softmax(dim=-1))

    def forward(self, x):
        #out = x
        #for module in self.main.children():
        #    x = module(x)
        #    out.append(x)
        return self.main(x)
