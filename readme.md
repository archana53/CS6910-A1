# Neural networks from scratch
## Assignment 1: Fundamentals of Deep Learning

We implement a neural network from scratch using numpy in python. The code is available in the ipython notebook named **assignment1.ipynb**. We here explain the code and how to reproduce results using it.

### Activation functions
We have implemented _sigmoid, tanh, ReLU,_ and _linear_ activation functions along with their gradient.

### Loss Function
We have implemented _cross entropy_ and _mean squared error_ loss and their gradients when applied with softmax function.

### Weight Initialization
We have implemented _random_ and _Xavier_ weight initialization methods.

### Optimizers
We have implemented _sgd_, _momentum_, _nesterov_, _rmsprop_, _adam_, and _nadam_ optimizers for training the neural network.


### Layer class
This class contains information of a layer in the neural network. One layer object can be created as

`new_layer = layer(neurons,input_neurons,activation="sigmoid",weight_init="xavier")`

In the above example neurons refer to the number of neurons in layer, input_neurons refer to the number of input neurons in the layer, activation refers to the activation function applied after this layer and weight_init refers to the weight initialisation method to be used for initialising the layer parameters. 

We can pass "sigmoid", "relu", "tanh", "linear" to activation for using different activation functions. If no value is passed, the activation is taken to be "sigmoid".

We can pass "random" or "xavier" to weight_init for using different initialisation methods. If nothing is passed then random initialisation is done for parameters.

### Models class
A model object can be created using the _models_ class. A model object can be initialised as follows:

`model = models(input_size,loss="mse",optimizer=opt,lamda=lamda)`

In the above example input_size refers to the input layer size given to the model, loss refers to the loss function used to train the model, optimizer refers to the optimizer type used for training and lamda refers to the weight decay parameter used for regularisation.

We can pass "sgd", "momentum", "nesterov", "rmsprop", "adam", and "nadam" to the optimizer for using different optimiser functions.

We can pass "mse" for mean squared error and "cross_entropy" for cross entropy loss to loss for using different loss functions.

### Adding a layer to model
The member function _add_layer_ of model class can be used to add a layer to the model. We can call it as follows

`model.add_layer(neurons,weight_init="random",activation="sigmoid")`

where neurons refer to the number of neurons in the layer.

### Training the model
The member function _train_ of model class can be used to train the model. An example to call it is

`model.train(xtrain,ytrain,val=(xval,yval),lr=0.01,batch_size=16,epochs=10)`

where xtrain and ytrain are input and output of training data, and xval, yval are input and output of validation data. lr referes to the learning rate, batch_size referes to the batch size used for training, and epochs is the number of epochs for training the model.

One thing to be noted here is that ytrain and yval are one hot encoded vectors of the output. We can transform the output into one hot encoded matrix by using the _one_hot_encoding_ function.

There are additional parameters which can be passed for different different optimizer function usage. These are _beta1_, _beta2_, and _gamma_ for adam, nadam, nesterov and momentum functions.

### Building a model

An example to build a model is shown below

`model = models(784,optimizer="rmsprop",loss="mse")
model.add_layer(32,activation="relu",weight_init="xavier")
model.add_layer(len(class_names),activation="relu",weight_init="xavier")
model.print()`

model.print() is a helper function to check the final model that we have built.

### Using the model

The member function _predict_ of model class can be used to predict the outputs on given inputs. It can be used as

`ytest = np.argmax(model.predict(xtest),axis=1)`


### Note
To run the model, the whole notebook needs to be run sequentially. There is a markdown cell in the notebook saying "Can put training blocks here" where we can build and test our own models. All the cells in the notebook need to be run till this point for the model to be trained.

### Wandb sweep

Last section in the notebook contains the wandb sweep code we used for different sweep configurations.