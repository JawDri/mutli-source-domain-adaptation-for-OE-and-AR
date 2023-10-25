#from easydl import *
from data import*
import os
from torchvision import models, utils
import torch.nn.functional as F
import torch.nn as nn
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
            nn.Linear(len(FEATURES), 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,256)
            )
    
    def forward(self, x):
        logits = self.linear_nn(x)
        return logits

model=Net()
class ResNet50Fc(BaseFeatureExtractor):
    def __init__(self,model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        print (normalize)
        if model_path:
            if os.path.exists(model_path):
                model_resnet = model
                # original saved file with DataParallel
                state_dict = torch.load(model_path)['state_dict']
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                #model.load_state_dict(new_state_dict)
                model_resnet.load_state_dict(new_state_dict)
            else:
                raise Exception('invalid model path!')
        else:
            model_resnet = model

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            
        else:
            self.normalize = False
       
        #model_resnet = self.model_resnet

        self.linear_nn = model_resnet.linear_nn
        #self.__in_features = model_resnet.fc.in_features

        # self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        #if self.normalize:
            #x = (x - self.mean) / self.std
       
        x = self.linear_nn(x)

        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return 256





class CLS(nn.Module):

    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, pretrain=False):
        super(CLS, self).__init__()
        self.pretrain = pretrain
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.fc = nn.Linear(bottle_neck_dim, out_dim)
            self.main = nn.Sequential(self.bottleneck,self.fc,nn.Softmax(dim=-1))
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(self.fc,nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out



class GeneratedNetwork(nn.Module):
    def __init__(self, vector_dim):
        super(GeneratedNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(vector_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, vector_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        y = self.main(x)
        return y


class GradientReverseLayer(torch.autograd.Function):
    """
    usage:(can't be used in nn.Sequential, not a subclass of nn.Module)::

        x = Variable(torch.ones(1, 2), requires_grad=True)
        grl = GradientReverseLayer()
        grl.coeff = 0.5
        y = grl(x)

        y.backward(torch.ones_like(y))

        print(x.grad)

    """
    def __init__(self):
        self.coeff = 1.0

    def forward(self, input):
        return input

    def backward(self, gradOutput):
        return -1.0 * gradOutput

class GradientReverseModule(nn.Module):
    """
    wrap GradientReverseLayer to be a nn.Module so that it can be used in ``nn.Sequential``

    usage::

        grl = GradientReverseModule(lambda step : aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

        x = Variable(torch.ones(1), requires_grad=True)
        ans = []
        for _ in range(10000):
            x.grad = None
            y = grl(x)
            y.backward()
            ans.append(variable_to_numpy(x.grad))

        plt.plot(list(range(10000)), ans)
        plt.show() # you can see gradient change from 0 to -1
    """
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.grl = GradientReverseLayer()

    def forward(self, x):
        coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        self.grl.coeff = coeff
        return self.grl.apply(x)

def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    '''
    change gradually from A to B, according to the formula (from <Importance Weighted Adversarial Nets for Partial Domain Adaptation>)
    A + (2.0 / (1 + exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)

    =code to see how it changes(almost reaches B at %40 * max_iter under default arg)::

        from matplotlib import pyplot as plt

        ys = [aToBSheduler(x, 1, 3) for x in range(10000)]
        xs = [x for x in range(10000)]

        plt.plot(xs, ys)
        plt.show()

    '''
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)

class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature, hidden_layer=True):
        super(AdversarialNetwork, self).__init__()
        if hidden_layer:
            self.main = nn.Sequential(
                nn.Linear(in_feature, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1),
                nn.Softmax()
            )
        else:
            self.main = nn.Sequential(
                nn.Linear(in_feature, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Softmax()
            )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x, grl=True):
        if grl:
            x_ = self.grl(x)
        y = self.main(x_)
        return y

def normalize_2d(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x

class Reconstructor(nn.Module):
    def __init__(self, in_dim):
        super(Reconstructor, self).__init__()
        self.init_size = 224 // 4
        self.l1 = nn.Sequential(nn.Linear(in_dim, 128 * self.init_size ** 2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, ),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def save_image_tensor(input_tensor: torch.Tensor, filename):
    # assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    utils.save_image(input_tensor, filename)