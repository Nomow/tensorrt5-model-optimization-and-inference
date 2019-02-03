import numpy as np
import pickle
import ctypes
from six.moves import cPickle


## Loads cifar 10 dataset from file
## @param file_name of cifar dataset
## @return raw_float_data - normalised raw img data in 1d array, labels - labels of each img
## @usage raw_data, labels = load_CIFAR10_dataset(file_name)
def load_CIFAR10_dataset(file_name):
    with open(file_name, 'rb') as f:                
        d = cPickle.load(f)
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
        data = d['data']
        labels = d['labels']
        raw_float_data = np.array(data, dtype=float) / 255.0
        return raw_float_data, labels
    
