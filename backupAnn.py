from __future__ import division
import numpy as np
import pickle, random, pdb, re
from PIL import Image

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
        self.Wxh = np.random.randn(384, 4) * 0.01 # Weight matrix for input to hidden
        self.Why = np.random.randn(4, 10) * 0.01 # Weight matrix for hidden to output
        self.bh = np.zeros((1, 4)) # hidden bias
        self.by = np.zeros((1, 10)) # output bias
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def _feed_forward(self, X):
        """
        Performs forward pass of the ANN
        :param X: input
        :return: hidden activations, softmax probabilities for the output
        """
        # Add code to calculate a1 and probs
        z1 = X.dot(self.Wxh) + self.bh
        a1 = np.tanh(z1)
        z2 = a1.dot(self.Why) + self.by
        exp_scores = np.exp(z2)
        probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        return a1, probs

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

    def _back_propagation(self, X, t, a1, probs,length):
        """
        Implementation of the backpropagation algorithm
        :param X: input
        :param t: target
        :param a1: hidden activation from forward pass
        :param probs: softmax probabilities of output from forward pass
        :return: dWxh, dWhy, dbh, dby
        """
        # Add code to compute the derivatives and return
        delta3 = probs
        delta3[range(length),t] -= 1
        dWhy = (a1.T).dot(delta3)
        dby = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(dWhy.T) * (1 - np.power(a1, 2))
        dWxh = np.dot(X.T, delta2)
        dbh = np.sum(delta2, axis=0)
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
            # Forward pass
            a1, probs = self._feed_forward(inputs)
            
            # Backpropagation
            dWxh, dWhy, dbh, dby = self._back_propagation(inputs, targets, a1, probs,len(inputs))

            # Perform the parameter update with gradient descent
            self.Wxh += -self.learning_rate * dWxh
            self.bh += -self.learning_rate * dbh
            self.Why += -self.learning_rate * dWhy
            self.by += -self.learning_rate * dby            
            

            # validation using the validation data

            validation_inputs = validation_data[0]
            validation_targets = validation_data[1]

            print 'Validation'

            # Forward pass
            a1, probs = self._feed_forward(validation_inputs)

            # Backpropagation
            dWxh, dWhy, dbh, dby = self._back_propagation(validation_inputs, validation_targets, a1, probs,len(validation_inputs))

            if regularizer_type == 'L2':
                dWhy = self.reg_lambda * self.Why
                dWxh = self.reg_lambda * self.Wxh

            # Perform the parameter update with gradient descent
            self.Wxh += -self.learning_rate * dWxh
            self.bh += -self.learning_rate * dbh
            self.Why += -self.learning_rate * dWhy
            self.by += -self.learning_rate * dby 

            if k%1 == 0:
                print "Epoch " + str(k) + " : Loss = " + str(self._calc_smooth_loss(loss, len(inputs), regularizer_type))

        #self.save('models.pkl')


    def predict(self, X, y):
        """
        Given an input X, emi
        :param X: test input
        :return: the output class
        """
        # Implement the forward pass and return the output class (argmax of the softmax outputs)
        a1, probs = self._feed_forward(X)
        
        hits = 0
        for i in xrange(len(y)):
            if np.where(probs[i]==max(probs[i]))[0][0] == y[i]: hits+=1

        print hits,len(X),hits*100/len(X)


    def save(self, model_file):
        """
        Saves the network to a file
        :param model_file: name of the file where the network should be saved
        :return: None
        """
        pickle.dump(self, open(model_file, 'wb'))

def to_one_hot(n):
    a = np.zeros(10)
    a[n] = 1
    return a

if __name__ == "__main__":
    """
    Digit recognition ANN
    """
    
    nn = NeuralNetwork(384,4,10)
    inputs = np.zeros(shape=(3000,384))
    targets = np.zeros(shape=(3000,10), dtype=np.int)
    
    pat = re.compile(r'\(\'(\d+.jpg)\',\s(\d)\)')
    labels_fd = open('labels.txt','r')
    labels = labels_fd.read()

    i=0
    for m in re.findall(pat,labels):
        img = Image.open('data16x8/'+m[0]).convert('RGB')
        arr = np.array(img)
        flat_arr = arr.ravel()
        inputs[i] = flat_arr
        targets[i] = int(m[1])
        targets.append(np.array([to_one_hot(m[1])]))
        i+=1
    inputs = np.array(inputs)
    targets = np.array(targets)

    nn.train(inputs, targets, (inputs[2100:2550], targets[2100:2550]), 1, regularizer_type='L2')
    #nn = load('models.pkl')
    nn.predict(inputs[2550:], targets[2550:])
