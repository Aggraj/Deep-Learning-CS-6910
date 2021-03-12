import numpy as np
import argparse
from scipy.special import expit
import tensorflow as tf
from sklearn.metrics import log_loss
from keras.datasets import fashion_mnist
output_classes = 10
inout_size = [784,output_classes]
non_linear_acts = ['softmax']  #Ouput has to be the probability

grad_layers  = []
grad_weights = []
grad_biases  = []
def command_line_args():
    import argparse
    parser = argparse.ArgumentParser(description='Model design')
    parser.add_argument('--num_hidden', help='No. of hidden layers', required=True,type=int)
    parser.add_argument('--sizes',help='Size of hidden layers', required=True,nargs='+',type=int)
    parser.add_argument('--activation', help='sigmoid/relu', required=True)
    parser.add_argument('--loss_func',help='cross_entropy', required = True)
    parser.add_argument('--opt',help='gd', required = True)

    args = vars(parser.parse_args()) #Converts to a dictionary
    print(args)
    return args
def sigmoid(x, varagin = False):
        if varagin == False:
            return expit(x)
        else:
            sigma_x = sigmoid(x)
            return sigma_x * (1 - sigma_x)

# def softmax(x, varagin = False):
#         if varagin == False:
#
#             out = np.zeros(x.shape)
#
#             for i in range(0, x.shape[1]):
#                 exps = np.exp(x[:, i])
#                 out[:, i] = exps / np.sum( exps)
#
#             return out
#         else:
#             pass
#
#         return
def softmax(x, varagin = False):
        if varagin == False:
            out = np.zeros(x.shape)
            for i in range(0, x.shape[1]):
                exps = np.exp(x[:, i] - np.max(x[:, i]))
                out[:, i] = exps / np.sum( exps)

            return out
            #return exps / np.sum(exps)
        else:
            pass

        return



def forward_go(train_images,weights,bias):
        pre_activation = []
        post_activation = []
        Inputlayer = []
        Outputlayer = []
        # train_images = X
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
        return(Outputlayer[-1],Inputlayer,Outputlayer)


def compute_grad(train_images,train_label, Inputlayer,Outputlayer,inout_size,weights,bias):

    grad_layers  = [0]*(len(inout_size) - 1)
    grad_weights = [0]*(len(weights))
    grad_biases  = [0]*(len(bias))
    n_samples = train_images.shape[0]
    mm = Outputlayer
    # print(mm[-1].shape)
    # print(mm[-1] - arr.T)
    # # print(mm[1].shape)
    # # print(mm[2].shape)
    # print(arr.shape)

    for layer in reversed(range(len(inout_size) - 1)):
        # print(layer)
    # for layer in reversed(range(4)):

                # -2 will make the indexing of layerInput/Output and layer indexing (remember you are counting input layer in n_layers) same
                if layer == len(inout_size) - 2:
                    grad_layers[layer] = Outputlayer[layer] - train_label.T
                    # grad_layers[layer] = Outputlayer[2] - arr.T
                    # grad_layers.append(mm[layer] - arr.T)


                    grad_weights[layer] = (1/n_samples) * np.matmul(grad_layers[layer], Outputlayer[layer-1].T )

                    grad_biases[layer] = (1/n_samples) * np.sum(grad_layers[layer], axis=1, keepdims = True)

                else:
                    if non_linear_acts[layer] == 'sigmoid':
                        grad_layers[layer] = (np.matmul(weights[layer+1].T, grad_layers[layer+1]))*(sigmoid(grad_layers[layer], varagin = True))


                    if layer == 0:
                        grad_weights[layer] = ((1/n_samples) * np.matmul(grad_layers[layer], train_imag))
                    else:
                        # grad_weights[layer] = (1/n_samples) * np.matmul(grad_layers[layer], self.layerOutput[layer-1].T)
                        # print(grad_layers[layer].shape)
                        # print(Outputlayer[layer-1].shape)
                        grad_weights[layer] = (1/n_samples) * np.matmul(grad_layers[layer], Outputlayer[layer-1].T)
                        # grad_weights[layer] = (1/n_samples) * np.matmul(grad_layers[layer], Inputlayer[layer-1])

                    grad_biases[layer] = (1/n_samples) * np.sum(grad_layers[layer], axis=1, keepdims = True)

    return (grad_layers, grad_biases, grad_weights)

def train_model(train_imag,train_label,train_val_images,train_val_labels,loss_function,opt,eta, max_epochs,weights,bias):

        train_loss = []
        val_loss= []

        t = 0
        for epoch in range(max_epochs):
            step = 0

            for num in range(0, train_imag.shape[0]):
            # for num in range(0,5):


                # predictions = self.forward_pass(X_train_mini)
                (output,Inputlayer,Outputlayer) = forward_go(train_imag,weights,bias)


                (grad_layers, grad_biases, grad_weights) = compute_grad(train_imag,train_label,Inputlayer,Outputlayer,inout_size,weights,bias)

                if opt == 'gd':
                    update_w = np.multiply(eta , grad_weights)
                    update_b = np.multiply(eta , grad_biases)


                t = t+1
                weights = weights - update_w
                bias = bias - update_b


                if loss_function == 'cross_entropy':
                    train_loss += log_loss(train_label.T,output)

            (output,Inputlayer,Outputlayer) = forward_go(train_imag,weights,bias)
            # (acc_train, correct_train,total_train) = self.evaluate(predictions_train.T, Y_train)
            # print('Ipsy')
            print(train_loss)
            # print('Epoch {0} Training Loss {1}'.format(epoch, train_loss[epoch]))



            (predictions_val,i,o) = forward_go(train_val_images,weights,bias)
            # Loss
            if loss_function == 'cross_entropy':
                val_loss.append(log_loss(train_val_labels.T, predictions_val))
                # print('Ipsy')


            # print('Epoch {0} Validation Loss {1}'.format(epoch, val_loss[epoch]))
            print(val_loss)


        return (train_loss, val_loss)

if __name__=='__main__':
    epochs = 5
    args = command_line_args()
    for i in args['sizes'] :
        inout_size.insert(1,i)
    for i in range(len(args['sizes'])):
        non_linear_acts.insert(0,args['activation'])
    weights = [np.random.randn(y, x) for x,y in zip(inout_size[:-1], inout_size[1:])]
    weights = np.array(weights)
    bias = [np.zeros((x, 1)) for x in inout_size[1:]]
    bias = np.array(bias)
    # pre_activation = []
    # post_activation = []
    # Inputlayer = []
    # Outputlayer = []
    # print(weights.shape)
    # print(inout_size[1:])
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # nval = 0.1*train_images.shape[0]
    # train_images = train_images[:50000,:]
    # train_labels = train_labels[:50000,]
    # train_val_image = train_images[50000:,:]
    # train_val_labels = train_labels[50000:,]
    n_samples = train_images.shape[0]
    train_images = train_images.reshape(n_samples,-1)
    train_imag = train_images[:50000,:]
    # train_label = train_labels[:50000,]
    train_val_images = train_images[50000:,:]
    # train_val_labels = train_labels[50000:,]
    # n_samples = train_imag.shape[0]
    labels = np.zeros((train_labels.shape[0],output_classes))
    for i in range(train_labels.shape[0]):
        e = [0.0]*output_classes
        e[train_labels[i]] = 1.0
        labels[i] = e
    train_label = labels[:50000,:]
    train_val_labels = labels[50000:,:]
    mean = train_imag.mean(axis=0)
    std  = train_imag.std(axis = 0)
    train_imag = (train_imag - mean)/std
    train_val_images = (train_val_images - mean)/std

    (output,Inputlayer,Outputlayer) = forward_go(train_imag,weights,bias)
    (grad_layers, grad_biases, grad_weights) = compute_grad(train_imag,train_label,Inputlayer,Outputlayer,inout_size,weights,bias)
    (tr_loss,val_loss) = train_model(train_imag,train_label,train_val_images,train_val_labels,args['loss_func'],args['opt'], 0.01, 5,weights,bias)
print(np.array(train_val_images).shape)
print(np.array(train_imag).shape)
print(np.array(train_val_labels).shape)
print(np.array(train_label).shape)
