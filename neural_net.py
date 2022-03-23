# version 1.4

from tkinter import NE, W
from typing import List
from wsgiref.handlers import BaseCGIHandler
import numpy as np

from operations import *

class NeuralNetwork():
    '''
    A class for a fully connected feedforward neural network (multilayer perceptron).
    :attr n_layers: Number of layers in the network
    :attr activations: A list of Activation objects corresponding to each layer's activation function
    :attr loss: A Loss object corresponding to the loss function used to train the network
    :attr learning_rate: The learning rate
    :attr W: A list of weight matrices. The first row corresponds to the biases.
    '''

    def __init__(self, n_features: int, layer_sizes: List[int], activations: List[Activation], loss: Loss,
                 learning_rate: float=0.01, W_init: List[np.ndarray]=None):
        '''
        Initializes a NeuralNetwork object
        :param n_features: Number of features in each training examples
        :param layer_sizes: A list indicating the number of neurons in each layer
        :param activations: A list of Activation objects corresponding to each layer's activation function
        :param loss: A Loss object corresponding to the loss function used to train the network
        :param learning_rate: The learning rate
        :param W_init: If not None, the network will be initialized with this list of weight matrices
        '''

        sizes = [n_features] + layer_sizes
        if W_init:
            assert all([W_init[i].shape == (sizes[i] + 1, sizes[i+1]) for i in range(len(layer_sizes))]), \
                "Specified sizes for layers do not match sizes of layers in W_init"
        assert len(activations) == len(layer_sizes), \
            "Number of sizes for layers provided does not equal the number of activations provided"

        self.n_layers = len(layer_sizes)
        self.activations = activations
        self.loss = loss
        self.learning_rate = learning_rate
        self.W = []
        for i in range(self.n_layers):
            if W_init:
                self.W.append(W_init[i])
            else:
                rand_weights = np.random.randn(sizes[i], sizes[i+1]) / np.sqrt(sizes[i])
                biases = np.zeros((1, sizes[i+1]))
                self.W.append(np.concatenate([biases, rand_weights], axis=0))

    def forward_pass(self, X) -> (List[np.ndarray], List[np.ndarray]):
        '''
        Executes the forward pass of the network on a dataset of n examples with f features. Inputs are fed into the
        first layer. Each layer computes Z_i = g(A_i) = g(Z_{i-1}W[i]).
        :param X: The training set, with size (n, f)
        :return A_vals: a list of a-values for each example in the dataset. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
                Z_vals: a list of z-values for each example in the dataset. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
        '''

        #####################################
        # YOUR CODE HERE
        #####################################

        nExamples = len(X)
        nFeatures = X.shape[1]

        def init_a_vals():
            A_vals = []
            for i_layer in range(self.n_layers):
                layer_size = len(self.W[i_layer][0])
                A_vals.append([[ 0 for i_neuron in range(layer_size) ] for i_example in range(nExamples)])
            return A_vals
        
        def init_z_vals():
            return [[] for i_layer in range(self.n_layers)]

        a_vals = init_a_vals()
        z_vals = init_z_vals()

        # base case
        i_layer = 0
        i_layer_size = len(self.W[i_layer][0])

        """
        print("X")
        print(X)
        print("Ws")
        print(self.W)
        print("n_layers")
        print(self.n_layers)
        """

        # base case
        for i_example in range(nExamples):
            # init a_values base case
            for i_neuron in range(i_layer_size):
                a_vals[i_layer][i_example][i_neuron] += self.W[i_layer][0][i_neuron] # dummy weight
                for i_feature in range(nFeatures):
                    a_vals[i_layer][i_example][i_neuron] += X[i_example, i_feature]*self.W[i_layer][i_feature+1][i_neuron]
                
            # init g_values base case
            z_vals[i_layer].append(self.activations[i_layer].value(a_vals[i_layer][i_example]))
        
        """
        print("a vals")
        print(a_vals)
        print("z vals")
        print(z_vals)

        print("a0 layer 2, example 0 should be")
        a_1_0_0 = self.W[i_layer+1][0][0] + (z_vals[i_layer][0][0] * self.W[i_layer+1][1][0] ) + (z_vals[i_layer][0][1]* self.W[i_layer+1][2][0]) + (z_vals[i_layer][0][2] * self.W[i_layer+1][3][0]) + (z_vals[i_layer][0][3] * self.W[i_layer+1][4][0])
        print(a_1_0_0)
        """
        # standard case
        i_layer += 1
        if i_layer != self.n_layers:
            

            while (i_layer < self.n_layers):
                i_layer_size = len(self.W[i_layer][0])
                nZValues = len(z_vals[i_layer - 1][0])
                for i_example in range(nExamples):
                    # init next a_values
                    for i_neuron in range(i_layer_size):
                        a_vals[i_layer][i_example][i_neuron] += self.W[i_layer][0][i_neuron] # dummy weight
                        for i_z in range(nZValues):
                            a_vals[i_layer][i_example][i_neuron] += z_vals[i_layer - 1][i_example][i_z]*self.W[i_layer][i_z+1][i_neuron]
                    # init next z_values
                    z_vals[i_layer].append(self.activations[i_layer].value(a_vals[i_layer][i_example]))
                i_layer += 1
        return a_vals, z_vals

        

    def backward_pass(self, A_vals, dLdyhat) -> List[np.ndarray]:
        '''
        Executes the backward pass of the network on a dataset of n examples with f features. The delta values are
        computed from the end of the network to the front.
        :param A_vals: a list of a-values for each example in the dataset. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :param dLdyhat: The derivative of the loss with respect to the predictions (y_hat), with shape (n, layer_sizes[-1])
        :return deltas: A list of delta values for each layer. There are n_layers items in the list and
                        each item is an array of size (n, layer_sizes[i])
        '''
  

        nExamples = len(A_vals[0])
        def init_deltas():
            deltas = []
            for i_layer in range(self.n_layers):
                layer_size = len(self.W[i_layer][i_layer])
                deltas.append([ [ 0 for i_neuron in range(layer_size) ] for i_example in range(nExamples) ])
            return deltas
        
        deltas = init_deltas()
        
        """

        print("A Vals")
        print(A_vals)
        print("DlDyhat")
        print(dLdyhat)
        print("weights")
        print(self.W)

        """

        # base case (last layer)
        i_layer = self.n_layers-1
        i_layer_size = len(self.W[i_layer][-1])
        for i_example in range(nExamples):
            neuron_derivatives = self.activations[i_layer].derivative(A_vals[i_layer][i_example])
            for i_neuron in range(i_layer_size):
                deltas[i_layer][i_example][i_neuron] = dLdyhat[i_example][i_neuron] * neuron_derivatives[i_neuron]

        """
        print("deltas after base")
        print()
        print(deltas)

        """

        # standard case
        i_layer -= 1
        if i_layer != -1:
            while i_layer > -1:
                i_layer_size = len(self.W[i_layer][0])
                for i_example in range(nExamples):
                    neuron_derivatives = self.activations[i_layer].derivative(A_vals[i_layer][i_example])
                    for i_neuron in range(i_layer_size):
                        i_previous_layer = i_layer+1
                        branches = len(self.W[i_previous_layer][0])
                        
                        for i_previous_neuron in range(branches):
                            prev_delta = deltas[i_previous_layer][i_example][i_previous_neuron]
                            weight = self.W[i_previous_layer][i_neuron][i_previous_neuron]
                            deltas[i_layer][i_example][i_neuron] += prev_delta*weight
                        deltas[i_layer][i_example][i_neuron] *= neuron_derivatives[i_neuron]
                i_layer -= 1

        """
        print('dir')
        print(self.activations[0].derivative(A_vals[0][0]))
        print("delta")
        delta0 = ((self.W[1][1][0]* deltas[1][0][0]) + (self.W[1][1][1]*deltas[1][0][1])) * self.activations[0].derivative(A_vals[0][0])[1]
        delt1 = ((self.W[1][0][0]* deltas[1][0][0]) + (self.W[1][0][1]*deltas[1][0][1])) * self.activations[0].derivative(A_vals[0][0])[0]
        print(delta0)
        print(delt1)

        # print deltas after
        print(deltas)
        """
        return deltas

    def update_weights(self, X, Z_vals, deltas) -> List[np.ndarray]:
        '''
        Having computed the delta values from the backward pass, update each weight with the sum over the training
        examples of the gradient of the loss with respect to the weight.
        :param X: The training set, with size (n, f)
        :param Z_vals: a list of z-values for each example in the dataset. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :param deltas: A list of delta values for each layer. There are n_layers items in the list and
                       each item is an array of size (n, layer_sizes[i])
        :return W: The newly updated weights (i.e. self.W)
        '''

        #####################################
        # YOUR CODE HERE
        #####################################



        return None

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> (List[np.ndarray], List[float]):
        '''
        Trains the neural network model on a labelled dataset.
        :param X: The training set, with size (n, f)
        :param y: The targets for each example, with size (n, 1)
        :param epochs: The number of epochs to train the model
        :return W: The trained weights
                epoch_losses: A list of the training losses in each epoch
        '''

        epoch_losses = []
        for epoch in range(epochs):
            A_vals, Z_vals = self.forward_pass(X)   # Execute forward pass
            y_hat = Z_vals[-1]                      # Get predictions
            L = self.loss.value(y_hat, y)           # Compute the loss
            print("Epoch {}/{}: Loss={}".format(epoch, epochs, L))
            epoch_losses.append(L)                  # Keep track of the loss for each epoch

            dLdyhat = self.loss.derivative(y_hat, y)         # Calculate derivative of the loss with respect to output
            deltas = self.backward_pass(A_vals, dLdyhat)     # Execute the backward pass to compute the deltas
            self.W = self.update_weights(X, Z_vals, deltas)  # Calculate the gradients and update the weights

        return self.W, epoch_losses

    def evaluate(self, X: np.ndarray, y: np.ndarray, metric) -> float:
        '''
        Evaluates the model on a labelled dataset
        :param X: The examples to evaluate, with size (n, f)
        :param y: The targets for each example, with size (n, 1)
        :param metric: A function corresponding to the performance metric of choice (e.g. accuracy)
        :return: The value of the performance metric on this dataset
        '''

        A_vals, Z_vals = self.forward_pass(X)       # Make predictions for these examples
        y_hat = Z_vals[-1]
        metric_value = metric(y_hat, y)     # Compute the value of the performance metric for the predictions
        return metric_value

