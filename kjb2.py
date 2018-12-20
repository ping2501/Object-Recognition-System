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

'''# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
'''

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 75
INIT_LR = 1e-3
BS = 32

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = glob.glob("C:\\Users\\KD\\PycharmProjects\\HandGestureRecognition\\dataset\\*\\*.jpg")

random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
labels = []
#labels=np.zeros(4852)

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (56,56))
	image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = img_to_array(image)
	data.append(image)
	l =  (imagePath.split(os.path.sep)[-2].split("_"))
	#labels = np.append(labels,int(l[0]))
	labels.append(int(l[0]))




labels = np_utils.to_categorical(labels, len(imagePaths))

data = np.array(data, dtype="float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
'''model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")
'''

model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='same',name='conv2_1', input_shape=(56,56,1)))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(len(imagePaths)))
model.add(Activation('softmax'))
#print (model.summary())

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")

'''H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
'''
print (trainX.shape)
print (trainY.shape)

H=model.fit(trainX, trainY, nb_epoch=3, batch_size=32,  verbose=1)

Res=model.predict_classes(testX)

i=0

for a in Res:
	print (a, testY[i])

# save the model to disk
print("[INFO] serializing network...")
model.save_weights("C:\\Users\\KD\\PycharmProjects\\HandGestureRecognition\\model1\\aa.h5")

json_string = model.to_json()
open("C:\\Users\\KD\\PycharmProjects\\HandGestureRecognition\\model1\\aa.json", 'w').write(json_string)
model.save_weights("C:\\Users\\KD\\PycharmProjects\\HandGestureRecognition\\model1\\aa.h5")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS


'''plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("/dataset")
'''