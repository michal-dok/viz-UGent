from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
import keract
import lime
from lime import lime_image
import numpy as np
from skimage.segmentation import slic
from PIL import Image
import base64
import io

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


def reduce_dim(dataset, components=2):
    """takes array of images (one row one image) and returns reduced dataset"""
    model = PCA(n_components=components)
    reduced = model.fit_transform(dataset)
    return reduced

# Function to convert image to base64 format for display in Dash
def convert_image_to_base64(image):
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def preprocess_images(images, nsamples=1):
    input_shape = (nsamples, 32, 32, 3) 
    img_array = images.reshape(input_shape)
    img_array = img_array / 255.0
    return img_array
