#from easydl import *
import numpy as np
import torch
from collections.abc import Iterable

import numpy as np

class Accumulator(dict):
    """
    accumulate data and store them in a dict

    usage::

        with Accumulator(['weight', 'coeff']) as accumulator:
            for data in data_generator():
                # forward ......
                weight = xxx
                coeff = xxx

                accumulator.updateData(scope=globals())

        # do whatever with accumulator['weight'] and accumulator['coeff']

    """

    def __init__(self, name_or_names, accumulate_fn=np.concatenate):
        super(Accumulator, self).__init__()
        self.names = [name_or_names] if isinstance(name_or_names, str) else name_or_names
        self.accumulate_fn = accumulate_fn
        for name in self.names:
            self.__setitem__(name, [])


    def updateData(self, scope):
        for name in self.names:
            self.__getitem__(name).append(scope[name])


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb:
            print(exc_tb)
            return False

        for name in self.names:
            self.__setitem__(name, self.accumulate_fn(self.__getitem__(name)))

        return True

class TrainingModeManager:
    """
    automatic set and reset net.train(mode)
    usage::

        with TrainingModeManager(net, train=True): # or with TrainingModeManager([net1, net2], train=True)
            do whatever
    """
    def __init__(self, nets, train=False):
        self.nets = nets if isinstance(nets, Iterable) else [nets]
        self.modes = [net.training for net in nets]
        self.train = train

    def __enter__(self):
        for net in self.nets:
            net.train(self.train)
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for (mode, net) in zip(self.modes, self.nets):
            net.train(mode)
        self.nets = None # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True

def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    '''
    change as initial_lr * (1 + gamma * min(1.0, iter / max_iter) ) ** (- power)
    as known as inv learning rate sheduler in caffe,
    see https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto

    the default gamma and power come from <Domain-Adversarial Training of Neural Networks>

    code to see how it changes(decays to %20 at %10 * max_iter under default arg)::

        from matplotlib import pyplot as plt

        ys = [inverseDecaySheduler(x, 1e-3) for x in range(10000)]
        xs = [x for x in range(10000)]

        plt.plot(xs, ys)
        plt.show()

    '''
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))
def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)

class Nonsense(object):
    """
    placeholder class to support a.b.c().e.f.g.h with nonsense value
    """

    def __getattr__(self, item):
        if item not in self.__dict__:
            self.__dict__[item] = Nonsense()
        return self.__dict__[item]

    def __call__(self, *args, **kwargs):
        return Nonsense()

    def __str__(self):
        return "Nonsense object!"

    def __repr__(self):
        return "Nonsense object!"

def normalize_weight(x, cut=0, expand=False):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val + 1e-10) / (max_val - min_val + 1e-10)
    if expand:
        x = x / torch.mean(x)
        # x = torch.where(x >= cut, x, torch.zeros_like(x))
    return x.detach()

def is_in_notebook():
    import sys
    return 'ipykernel' in sys.modules

def clear_output():
    """
    clear output for both jupyter notebook and the console
    """
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    if is_in_notebook():
        from IPython.display import clear_output as clear
        clear()

def l2_norm(input, dim=1):
    norm = torch.norm(input,dim=dim,keepdim=True)
    output = torch.div(input, norm)
    return output


def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


class OptimWithSheduler:
    """
    usage::

        op = optim.SGD(lr=1e-3, params=net.parameters()) # create an optimizer
        scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=100, power=1, max_iter=100) # create a function
        that receives two keyword arguments:step, initial_lr
        opw = OptimWithSheduler(op, scheduler) # create a wrapped optimizer
        with OptimizerManager(opw): # use it as an ordinary optimizer
            loss.backward()
    """
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1

def variable_to_numpy(x):
    """
    convert a variable to numpy, avoid too many parenthesis
    if the variable only contain one element, then convert it to python float(usually this is the test/train/dev accuracy)
    :param x:
    :return:
    """
    ans = x.cpu().data.numpy()
    if torch.numel(x) == 1:
        # make sure ans has no shape. (float requires number rather than ndarray)
        return float(np.sum(ans))
    return ans




class AccuracyCounter:
    """
    in supervised learning, we often want to count the test accuracy.
    but the dataset size maybe is not dividable by batch size, causing a remainder fraction which is annoying.
    also, sometimes we want to keep trace with accuracy in each mini-batch(like in train mode)
    this class is a simple class for counting accuracy.

    usage::

        counter = AccuracyCounter()
        iterate over test set:
            counter.addOntBatch(predict, label) -> return accuracy in this mini-batch
        counter.reportAccuracy() -> return accuracy over whole test set
    """
    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0

    def addOneBatch(self, predict, label):
        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal

    
    def reportAccuracy(self):
        """
        :return: **return nan when 0 / 0**
        """
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float)
