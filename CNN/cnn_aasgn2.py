from keras.models import Sequential
#import tensorflow as tf
#import tensorflow.keras as keras
#from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer
from keras.layers import Dense, Conv2D, Flatten
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import os
from keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow import keras
import scipy.io as sio
from mat4py import savemat
import tensorflow
from tensorflow.keras.layers import Dropout
from keras.models import Model
import pickle
import numpy as np
import argparse
from scipy.special import expit
import tensorflow as tf
from sklearn.metrics import log_loss
from keras.datasets import fashion_mnist
output_classes = 10

def command_line_args():
    import argparse
    parser = argparse.ArgumentParser(description='Model design')
    parser.add_argument('--num_filter', help='Number of filters ', required=True,nargs='+',type=int)
    parser.add_argument('--filter_size',help='Size of filters', required=True,nargs='+',type=int)
    parser.add_argument('--activation', help='sigmoid/relu', required=True)
    parser.add_argument('--num_neurons',help='Number_of_neurons', required = True)
    args = vars(parser.parse_args()) #Converts to a dictionary
    print(args)
    return args

if __name__=='__main__':
    num_epoch = 5
    args = command_line_args()
    print(args['num_filter'][4])
    # xTrain, xTest, yTrain, yTest = train_test_split(data0, data1, test_size = 0.2, random_state = 0)
    pickle_in = open("train_x.pickle","rb")
    xTrain = pickle.load(pickle_in)

    pickle_in = open("train_labels.pickle","rb")
    yTrain = pickle.load(pickle_in)
    ll = []
    print(np.array(yTrain).shape)
    # for ii in yTrain:
    #     ll.append(ii)
    # print(np.array(ll).shape)
    # print(yTrain[0])
    model = Sequential()
    # # model.add(Conv2D(args['num_filter'][0], args['num_filter'][0], activation=args['activation'], input_shape=(4,7,1)))
    model.add(Conv2D(args['num_filter'][0], args['filter_size'][0], activation=args['activation'], input_shape=(200,200,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(args['num_filter'][1],args['filter_size'][1], activation=args['activation']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(args['num_filter'][2], args['filter_size'][2], activation=args['activation']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(args['num_filter'][3], args['filter_size'][3], activation=args['activation']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(args['num_filter'][4],args['filter_size'][4], activation=args['activation']))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(args['num_neurons'], activation=args['activation']))
    model.add(Dense(10))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    model.fit(xTrain, yTrain, batch_size=64,epochs=num_epoch, verbose=1)

# ypred = model.predict(xtest)
# print(model.evaluate(xtrain, ytrain))
# print("MSE: %.4f" % mean_squared_error(ytest, ypred))

# x_ax = range(len(ypred))
# plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
# plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
# plt.legend()
# plt.show()
