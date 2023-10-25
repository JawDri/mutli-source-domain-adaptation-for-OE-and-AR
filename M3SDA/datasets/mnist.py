import numpy as np
from scipy.io import loadmat
import pandas as pd




base_dir = './data'
def load_mnist(scale=True, usps=False, all_use=False):
    Target_train = pd.read_csv("./data/Target_train.csv")
    Target_test = pd.read_csv("./data/Target_test.csv")

    Target_train_data = Target_train.drop(['labels'], axis= 1).values
    Target_train_labels = Target_train.labels.values

    Target_test_data = Target_test.drop(['labels'], axis= 1).values
    Target_test_labels = Target_test.labels.values


    #train_label = np.argmax(Target_train_labels, axis=1)
    
    #test_label = np.argmax(Target_test_labels, axis=1)
    
    
    '''Target_train_data = Target_train_data[:765]
    Target_train_labels = Target_train_labels[:765]
    Target_test_data = Target_test_data[:495]
    Target_test_labels = Target_test_labels[:495]'''

    return Target_train_data.astype(np.float32), Target_train_labels, Target_test_data.astype(np.float32), Target_test_labels
