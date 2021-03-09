import numpy as np
import argparse
from scipy.special import expit
import tensorflow as tf
from keras.datasets import fashion_mnist
output_classes = 10
inout_size = [784,output_classes]
non_linear_acts = ['softmax']  #Ouput has to be the probability

def command_line_args():
    import argparse
    parser = argparse.ArgumentParser(description='Model design')
    parser.add_argument('--num_hidden', help='No. of hidden layers', required=True,type=int)
    parser.add_argument('--sizes',help='Size of hidden layers', required=True,nargs='+',type=int)
    parser.add_argument('--activation', help='sigmoid/relu', required=True)
    parser.add_argument('--loss_func',help='cross_entropy', required = True)
    args = vars(parser.parse_args()) #Converts to a dictionary
    print(args)
    return args
def sigmoid(x, varagin = False):
        if varagin == False:
            return expit(x)
        else:
            sigma_x = self.sigmoid(x)
            return sigma_x * (1 - sigma_x)

def softmax(x, varagin = False):
        if varagin == False:

            out = np.zeros(x.shape)

            for i in range(0, x.shape[1]):
                exps = np.exp(x[:, i])
                out[:, i] = exps / np.sum( exps)

            return out
        else:
            pass

        return
if __name__=='__main__':
    args = command_line_args()
    for i in args['sizes'] :
        inout_size.insert(1,i)
    for i in range(len(args['sizes'])):
        non_linear_acts.insert(0,args['activation'])
    weights = [np.random.randn(y, x) for x,y in zip(inout_size[:-1], inout_size[1:])]
    weights = np.array(weights)
    bias = [np.zeros((x, 1)) for x in inout_size[1:]]
    bias = np.array(bias)
    pre_activation = []
    post_activation = []
    Inputlayer = []
    Outputlayer = []
    # print(weights.shape)
    # print(inout_size[1:])
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    n_samples = train_images.shape[0]
    train_images = train_images.reshape(n_samples,-1)
    # print(M.shape)
    # print(n_samples)
    # n_samples = train_images.shape[0]
    dim = train_images.shape[1]
    for layer in range(0,len(inout_size) - 1):
                if layer == 0:
                    pre_activation.append(np.matmul(weights[layer], train_images.T) + bias[layer])

                else:
                    pre_activation.append(np.matmul(weights[layer], post_activation[layer - 1]) + bias[layer])


                Inputlayer.append(pre_activation[layer])

                if non_linear_acts[layer] == 'sigmoid':
                    post_activation.append(sigmoid(pre_activation[layer]))

                elif (non_linear_acts[layer] == 'softmax') and (layer == len(inout_size) - 2) :
                    post_activation.append(softmax(pre_activation[layer]))

                Outputlayer.append(post_activation[layer])

print(Outputlayer[-1])
