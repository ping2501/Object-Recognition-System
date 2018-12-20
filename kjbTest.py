# USAGE
# python train.py --dataset dataset --model fashion.model --labelbin mlb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop
from keras.callbacks import Callback
import glob

#from pyimagesearch.smallervggnet import SmallerVGGNet

import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 3
INIT_LR = 1e-3
BS = 20

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = glob.glob("ADD YOUR OWN PATH\\t\\*.jpg")

random.seed(42)


# initialize the data and labels
data = []
labels = []
#labels=np.zeros(4852)

for imagePath in imagePaths:
	print (imagePath)
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28,28))
	image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = img_to_array(image)
	data.append(image)


data = np.array(data, dtype="float") / 255.0

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

model = model_from_json(open("ADD YOUR OWN PATH\\model7\\aa.json").read())
model.load_weights('ADD YOUR OWN PATH\\model7\\aa.h5')
# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print ("compiled")

Res = model.predict_classes(data)

i=0

print (Res)
