import sklearn
import plotly
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dataset_path = "./cifar-10-python.tar.gz"

data1 = unpickle("data_batch_1")
print(data1)