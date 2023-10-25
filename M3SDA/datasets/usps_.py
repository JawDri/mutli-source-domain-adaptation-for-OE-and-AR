from scipy.io import loadmat
import numpy as np
import sys
import pandas as pd
sys.path.append('../utils/')
from utils.utils import dense_to_one_hot
base_dir = './data'

def load_usps():
    Source_test = pd.read_csv("./data/Source_test_2.csv")
    Source_train = pd.read_csv("./data/Source_train_2.csv")

    Source_train_data = Source_train.drop(['labels'], axis= 1).values
    Source_train_labels = Source_train.labels.values

    Source_test_data = Source_test.drop(['labels'], axis= 1).values
    Source_test_labels = Source_test.labels.values



    #print('svhn train y shape before dense_to_one_hot->', Source_train_labels.shape)
    svhn_label = dense_to_one_hot(Source_train_labels)
    #print('svhn train y shape after dense_to_one_hot->',svhn_label.shape)
    
    svhn_label_test = dense_to_one_hot(Source_test_labels)
    
    '''Source_train_data = Source_train_data[:1584]
    svhn_label = svhn_label[:1584]
    Source_test_data = Source_test_data[:495]
    svhn_label_test = svhn_label_test[:495]'''

    return Source_train_data.astype(np.float32), svhn_label, Source_test_data.astype(np.float32), svhn_label_test

