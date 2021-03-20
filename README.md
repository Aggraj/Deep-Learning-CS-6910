# Deep-Learning-CS-6910

Assignment - 1 : Training a feed forward neuaral network from scratch using numpy. The training dataset is fashion MNIST. 

The code takes in : 

                    n_inputs  : number of neurons in the input layer

                    n_hidden  : size of the hidden layer with the number of neurons in each layer
                    
                    n_outputs : number of output neurons 
                    
                    loss_function : cross_entropy or mean squared error
                    
                    activation function : sigmoid, tanh, ReLu and Leaky ReLu 

The gradients can be calulated using different algorithms : 
1. gd - gradient descent 
2. mgd - momentum based gradient descent
3. ngd - Nesterov based accelerated gradient descent 
4. rmsprop
5. adam
6. nadam

Hyperparameter tune can be done using wandb - Hyperparameter_sweep.pynb




