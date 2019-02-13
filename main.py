from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Dense, TimeDistributed, Input
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.applications import VGG16

#from vgg16 import VGG16
from RoiPooling import RoiPooling
from get_regions import rmac_regions, get_size_vgg_feat_map

import scipy.io
import numpy as np
import utils
import rmac
import os

import cv2 as cv
from sklearn.metrics.pairwise import cosine_similarity

def loadImage(filename):
    # Load sample image
    file = utils.DATA_DIR_IMG + filename
    img = image.load_img(file)
    return img

def holiday_images():
    for filename in os.listdir(utils.DATA_DIR_IMG):
        img = loadImage(filename)
        if img is not None:
            RMAC = getRMAC(img)
            utils.save_obj(RMAC, filename)

def get_descriptors():
    descriptors = []
    for filename in os.listdir(utils.SAVE_DIR):
        descriptors.append((filename, utils.load_obj(filename)))
    return descriptors

def similarity(des1, des2):
    return float(cosine_similarity(des1, des2)[0][0])

def findBestMatches(query, data):
    result = []
    for filename, des in data:
        result.append([filename, similarity(query[1], des)])
    return sorted(result, key = lambda x: x[1], reverse=True) 


def getRMAC(img):

    # Resize
    scale = utils.IMG_SIZE / max(img.size)
    new_size = (int(np.ceil(scale * img.size[0])), int(np.ceil(scale * img.size[1])))
    img = img.resize(new_size)

    # Mean substraction
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_image(x)

    # Load RMAC model
    Wmap, Hmap = get_size_vgg_feat_map(x.shape[3], x.shape[2])
    regions = rmac_regions(Wmap, Hmap, 3)
    model = rmac.rmac((x.shape[1], x.shape[2], x.shape[3]), len(regions))
    # Compute RMAC vector
    print('Extracting RMAC from image...')
    RMAC = model.predict([x, np.expand_dims(regions, axis=0)])

    return RMAC

descriptors = get_descriptors()

query = descriptors.pop(4)
result = findBestMatches(query, descriptors)
print(query[0])
print(result)
print(result[0][1])