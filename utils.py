import pickle

DATA_DIR = 'data/'
WEIGHTS_FILE = 'vgg16_weights_th_dim_ordering_th_kernels.h5'
PCA_FILE = 'PCAmatrices.mat'
IMG_SIZE = 1024
SAVE_DIR = 'data/descriptors/'
DATA_DIR_IMG = 'data/images/'

import numpy as np
def save_obj(obj, filename):
    f = open(SAVE_DIR + filename, 'wb')
    pickle.dump(obj, f)
    f.close()
#    print("Object saved to %s " +  filename)


def load_obj(filename):
    f = open(SAVE_DIR + filename, 'rb')
    
    obj = np.load(f)
    f.close()
#    print("Object loaded to %s " +  filename)
    return obj


def preprocess_image(x):

    # Substract Mean
    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68

    # 'RGB'->'BGR'
    x = x[:, ::-1, :, :]

    return x