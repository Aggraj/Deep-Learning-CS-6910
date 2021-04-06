import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pdb
from tqdm import tqdm
import random
import pickle
from keras.utils.np_utils import to_categorical


train_dir = "/home/aggraj/Downloads/inaturalist_12K/train"
val_dir = "/home/aggraj/Downloads/inaturalist_12K/val"

CATEGORIES = ["Amphibia", "Animalia", "Arachnida", "Aves", "Fungi", "Insecta", "Mammalia", "Mollusca", "Plantae", "Reptilia"]

training_data = []
validation_data = []
IMG_SIZE = 200
train_x = []
train_labels = []
val_x = []
val_labels = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(train_dir,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        # mm[class_num] = 1.0
        # labels[class_num] = e

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                IMG_SIZE = 200
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array,class_num])  # add this to our training_data
            except Exception as mm:  # in the interest in keeping the output clean...
                pass

def create_validation_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(val_dir,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        # mm[class_num] = 1.0
        # labels[class_num] = e

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                IMG_SIZE = 200
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                validation_data.append([new_array,class_num])  # add this to our training_data
            except Exception as mm:  # in the interest in keeping the output clean...
                pass



create_training_data()
create_validation_data()
random.shuffle(training_data)
random.shuffle(validation_data)
print(np.array(training_data).shape)
print(np.array(validation_data).shape)

for features,label in training_data:
    train_x.append(features)
    train_labels.append(label)


for features,label in validation_data:
    val_x.append(features)
    val_labels.append(label)

train_x = np.array(train_x)
print(train_x.shape)


#pdb.set_trace()
print(np.array(train_labels).shape)
xx = np.array(train_labels)
yy = np.array(val_labels)
print(xx.shape)
train_labels = to_categorical(xx)
val_labels = to_categorical(yy)
# pdb.set_trace()
print(train_labels.shape)
print(val_labels.shape)
## xx = np.array(xx)
## print(xx.shape)
## for ii in range(9999):
#    # ff.append(np.array(train_labels[ii]))
## print(train_x[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
#
train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
val_x = np.array(val_x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#
#
## print(train_labels[0])
## print(train_labels[1])
## print(train_labels[9998].shape)
## print(arr[0])
## print(arr[1])
## Not working
## print(train_labels[1][9998])
#
## print(val_labels[0])
print(train_x.shape)
print(val_x.shape)
## print(np.array(train_labels[2]).shape)
## print(np.array(ff).shape)
pickle_out = open("train.pickle","wb")
pickle.dump(train_x, pickle_out)
pickle_out.close()

pickle_out = open("train_label.pickle","wb")
pickle.dump(train_labels, pickle_out)
pickle_out.close()

pickle_out = open("val.pickle","wb")
pickle.dump(val_x, pickle_out)
pickle_out.close()

pickle_out = open("val_label.pickle","wb")
pickle.dump(val_labels, pickle_out)
pickle_out.close()
