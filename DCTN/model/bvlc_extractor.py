import torch
import torch.nn as nn
import os



class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm1d):
                module.train(False)
            else:
                module.train(mode)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_nn = nn.Sequential(
            nn.Linear(32, 512),###
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,256)
            )

    def forward(self, x):
        logits = self.linear_nn(x)
        return logits


model=Net()
class Extractor(BaseFeatureExtractor):
    def __init__(self,model_path=None, normalize=True):
        super(Extractor, self).__init__()
        print (normalize)
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = model
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_resnet = model



        model_resnet = self.model_resnet

        self.linear_nn = model_resnet.linear_nn
        #self.__in_features = model_resnet.fc.in_features

        # self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):


        x = self.linear_nn(x)

        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return 256
