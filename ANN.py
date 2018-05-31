import numpy as np
import pickle, random, pdb, math
from PIL import Image
import re

def load(model_file):
    """
    Loads the network from the model_file
    :param model_file: file onto which the network is saved
    :return: the network
    """
    return pickle.load(open(model_file))

class NeuralNetwork(object):
    """
    Implementation of an Artificial Neural Network
    """
    def __init__(self, input_dim, hidden_size, output_dim, learning_rate=0.01, reg_lambda=0.01):
        """
        Initialize the network with input, output sizes, weights, biases, learning_rate and regularization parameters
        :param input_dim: input dim
        :param hidden_size: number of hidden units
        :param output_dim: output dim
        :param learning_rate: learning rate alpha
        :param reg_lambda: regularization rate lambda
        :return: None
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(self.hidden_size, self.input_dim) * 0.01 # Weight matrix for input to hidden
        self.Why = np.random.randn(self.output_dim, self.hidden_size) * 0.01 # Weight matrix for hidden to output
        self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.by = np.zeros((self.output_dim, 1)) # output bias
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.smooth_loss = -np.log(1.0/10) # loss at iteration 0

    def feed_forward(self, X):
        """
        Performs forward pass of the ANN
        :param X: input
        :return: hidden activations, softmax probabilities for the output
        """
        # xs, hs, ys, ps = None,None,N
        hs = [sigmoid(i) for i in np.dot(self.Wxh, X.T)+self.bh]
        ys = np.dot(self.Why, hs) + self.by
        ps = np.exp(ys) / np.sum(np.exp(ys))
        return hs,ps

    def _regularize_weights(self, dWhy, dWxh, Why, Wxh):
        """
        Add regularization terms to the weights
        :param dWhy: weight derivative from hidden to output
        :param dWxh: weight derivative from input to hidden
        :param Why: weights from hidden to output
        :param Wxh: weights from input to hidden
        :return: dWhy, dWxh
        """
        # Add code to calculate the regularized weight derivatives
        return dWhy, dWxh

    def _update_parameter(self, dWxh, dbh, dWhy, dby):
        """
        Update the weights and biases during gradient descent
        :param dWxh: weight derivative from input to hidden
        :param dbh: bias derivative from input to hidden
        :param dWhy: weight derivative from hidden to output
        :param dby: bias derivative from hidden to output
        :return: None
        """
        # Add code to update all the weights and biases here

    def back_propagation(self, xs, target, hs, ps):
        """
        Implementation of the backpropagation algorithm
        :param X: input
        :param t: target
        :param h_a: hidden activation from forward pass
        :param probs: softmax probabilities of output from forward pass
        :return: dWxh, dWhy, dbh, dby
        """
        dWxh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Why)        
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        dby = (ps-target)*sigmoid(ps)*(1-sigmoid(ps))
        dWhy = np.dot(dby,hs.T)

        dbh = (self.Wxh.T,dby)*sigmoid(ps)*(1-sigmoid(ps))
        dWxh = np.dot(dbh,xs.T)

        #for dparam in [dWxh, dWhy, dbh, dby]:
        #    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return dWxh, dWhy, dbh, dby

    def _calc_smooth_loss(self, loss, len_examples, regularizer_type=None):
        """
        Calculate the smoothened loss over the set of examples
        :param loss: loss calculated for a sample
        :param len_examples: total number of samples in training + validation set
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: smooth loss
        """
        if regularizer_type == 'L2':
            # Add regulatization term to loss

            return 1./len_examples * loss
        else:
            return 1./len_examples * loss

    def train(self, inputs, targets, validation_data, num_epochs, regularizer_type=None):
        """
        Trains the network by performing forward pass followed by backpropagation
        :param inputs: list of training inputs
        :param targets: list of corresponding training targets
        :param validation_data: tuple of (X,y) where X and y are inputs and targets respectively
        :param num_epochs: number of epochs for training the model
        :param regularizer_type: type of regularizer like L1, L2, Dropout
        :return: None
        """
        for k in xrange(num_epochs):
            loss = 0
            for i in xrange(len(inputs)):
                # Forward pass
                hs, ps = nn.feed_forward(inputs[i])

                # Backpropagation
                dWxh, dWhy, dbh, dby = nn.back_propagation(inputs[i], targets[i], hs, ps)

                # Perform the parameter update with gradient descent
                loss = -np.log(ps[i]) # softmax (cross-entropy loss)
                #smooth_loss = smooth_loss * 0.999 + loss * 0.001
                # perform parameter update with Adagrad
                for param, dparam in zip([self.Wxh, self.Why, self.bh, self.by], 
                                            [dWxh, dWhy, dbh, dby]):
                    param += -learning_rate * dparam / np.sqrt(1e-8) # adagrad update


            # # validation using the validation data

            # validation_inputs = validation_data[0]
            # validation_targets = validation_data[1]

            # print 'Validation'

            # for i in xrange(len(validation_inputs)):
            #     # Forward pass
            #     hs, ps = nn.feed_forward(inputs[i])

            #     # Backpropagation
            #     dWxh, dWhy, dbh, dby = nn.back_propagation(inputs[i], targets[i], hs, ps)

            #     if regularizer_type == 'L2':
            #         # Add code for regularization of weights


            #     # Perform the parameter update with gradient descent
            #     loss = -np.log(ps[i]) # softmax (cross-entropy loss)

    
            # if k%1 == 0:
            #     print "Epoch " + str(k) + " : Loss = " + str(self._calc_smooth_loss(loss, len(inputs), regularizer_type))


    def predict(self, X, targets):
        """
        Given an input X, emi
        :param X: test input
        :return: the output class
        """
        # Implement the forward pass and return the output class (argmax of the softmax outputs)

        for i in xrange(len(X)):
            hs, ps = self.feed_forward(X[i])

    def save(self, model_file):
        """
        Saves the network to a file
        :param model_file: name of the file where the network should be saved
        :return: None
        """
        pickle.dump(self, open(model_file, 'wb'))

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def to_one_hot(n):
    a = np.zeros(10)
    a[n] = 1
    return a



if __name__ == "__main__":
    """
    Toy problem where input = target
    """
    nn = NeuralNetwork(384,8,10)
    inputs = []
    targets = []
    DS = 3000

    pat = re.compile(r'\(\'(\d+.jpg)\',\s(\d)\)')
    labels_fd = open('labels.txt','r')
    labels = labels_fd.read()

    for m in re.findall(pat,labels):
        img = Image.open('data16x8/'+m[0]).convert('RGB')
        arr = np.array(img)
        shape = arr.shape
        flat_arr = arr.ravel()
        vector = np.matrix(flat_arr)
        inputs.append(vector)
        targets.append(to_one_hot(m[1]))
    nn.train(inputs[:2100], targets[:2100], (inputs[2100:2550], targets[2100:2550]), 1, regularizer_type='L2')
    #nn.predict(inputs[.85*DS:], targets[.85*DS:])
