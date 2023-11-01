from sklearn.decomposition import PCA
import plotly
import numpy as np
import matplotlib.pyplot as plt


def unpickle(file):
    """load dataset"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def row2array(row):
    """transforms one row of a dataset into an array W * H * 3 array (so that it can be visualised using plt.imshow)"""
    r, g, b = np.array_split(row, 3)    
    r_mat = r.reshape((32, 32))
    g_mat = g.reshape((32, 32))
    b_mat = b.reshape((32, 32))
    arr = np.dstack([r_mat, g_mat, b_mat])
    return arr

def row2array_gray(row):
    gray = row.reshape((32, 32))
    arr = gray
    return arr


def reduce_dim(dataset, components=2):
    """takes array of images (one row one image) and returns reduced dataset"""
    model = PCA(n_components=components)
    reduced = model.fit_transform(dataset)
    return reduced


dataset_path = "./data/cifar-10-python.tar.gz"

data1 = unpickle("./data/data_batch_1")

images = data1[b'data']
labels = data1[b'labels']

