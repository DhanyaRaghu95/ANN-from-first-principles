from __future__ import division
import numpy as np
import pickle, random, pdb, re
from PIL import Image



import numpy as np
import pickle


def to_one_hot(n):
    a = np.zeros(10)
    a[n] = 1
    return a

def load(model_file):
    """
    Loads the network from the model_file
    :param model_file: file onto which the network is saved
    :return: the network
    """
    return pickle.load(open(model_file))

class NeuralNetworks(object):
	"""Implementation of Artificial Neural Networks"""
	def __init__(self, input_dim, hidden_size, output_dim, learning_rate=0.01, reg_lambda=0.01):
		"""Initialize the network Parameters"""
		self.input_dim = input_dim
		self.hidden_size = hidden_size
		self.output_dim = output_dim
		self.Wxh = np.random.randn(self.hidden_size, self.input_dim) * 0.01
		self.Why = np.random.randn(self.output_dim, self.hidden_size) * 0.01
		self.bh = np.zeros((self.hidden_size, 1))
		self.by = np.zeros((self.output_dim, 1))
		self.learning_rate = learning_rate
		self.reg_lambda = reg_lambda

	def feed_forward(self, X):
		"""Performs the forward pass of the ANN
		Return the energy of activation of neurons in hidden and output layer
		"""
		h_a = sigmoid(np.dot(self.Wxh, X) + self.bh)
		probs = sigmoid(np.dot(self.Why, h_a) + self.by)
		return h_a, probs

	def regularize_weights(self, dWhy, dWhx, Why, Wxh):
		dWxh += self.reg_lambda + Wxh
		dWhy += self.reg_lambda + Why
		return dWxh, dWhy

	def backpropagation(self, X, t, h_a, probs):
		"""Performs the backpropagation of the ANN
		"""
		z1 = np.dot(self.Wxh, X) + self.bh
		z2 = np.dot(self.Why, sigmoid(z1)) + self.by

		delta = self.error_function(X, t) * sigmod_prime(z2)

		# Compute the derivatives associated with the weights and biases of the output layer
		delta_bo = delta
		delta_wo = np.dot(delta, sigmoid(z1).transpose())

		# Back propagate, compute derivatives for hidden layer
		sp = sigmoid_prime(z1)
		delta = np.dot(self.Why.transpose(), delta) * sp
		delta_bh = delta
		delta_wh = np.dot(delta, X.transpose())

		return delta_wh, delta_wo, delta_bh, delta_bo

	def error_function(self, output_activations, y):
		return (output_activations - y)

	def update_parameters(self, dWxh, dWhy, dbh, dby):
		self.Wxh = self.Wxh - self.learning_rate * dWxh
		self.Why = self.Why - self.learning_rate * dWhy
		self.bh = self.bh - self.learning_rate * dbh
		self.by = self.by - self.learning_rate * dby

	def calc_smooth_loss(self, loss, len_examples, regularizer_type=None):
		if regularizer_type == 'L2':
			loss = loss + 5 / (2*len_examples) * sum(self*Wxh ** 2)
		return 1.0 / len_examples * loss

	def train(self, inputs, targets, validation_data, num_epochs, regularizer_type = None):
		"""Trains the network"""
		for k in xrange(num_epochs):
			loss = 0
			for i in xrange(len(inputs)):
				# Perform the Forward Pass
				h_a, probs = self.feed_forward(inputs)

				# Perform the Backpropagation
				dWxh, dWhy, dbh, dby = self.backpropagation(inputs, targets, h_a, probs)

				# Regularize the weights if regularization is present
				if regularizer_type == 'L2':
					dWhy, dWxh = self.regularize_weights(dWhy, dWxh, self.Why, self.Wxh)

				# Perform the parameter update with gradient descent
				self.update_parameters(dWxh, dWhy, dbh, dby)

	def predict(self, test_data):
		"""Predict the accuracy of classification of the network
		Accuracy = (correctly classified instances) / total number of instances
		"""
		test_results = [(np.argmax(self.feed_forward(x)), y) for (x,y) in test_data]
		return sum(int(x==y) for (x,y) in test_results)

# Sigmoid function
def sigmoid(z):
	return 1.0 / 1.0 + np.exp(-z)

# Delta sigmoid function
def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))

if __name__ == "__main__":
    nn = NeuralNetworks(768,4,10)
    inputs = np.zeros(shape=(3000,768))
    targets = np.zeros(shape=(3000,10), dtype=np.int)

    # print targets;exit(0)
    
    pat = re.compile(r'\(\'(\d+.jpg)\',\s(\d)\)')
    labels_fd = open('labels.txt','r')
    labels = labels_fd.read()

    i=0
    for m in re.findall(pat,labels):
        img = Image.open('data16x16/'+m[0]).convert('RGB')
        arr = np.array(img)
        flat_arr = arr.ravel()
        inputs[i] = flat_arr
        # targets[0][i] = int(m[1])
        targets[i] = to_one_hot(int(m[1]))
        i+=1
    #print(len(targets),(targets[0]))
    #print(len(inputs),(inputs[0]))
    nn.train(inputs[:2100],targets[:2100], (inputs[2100:2550], targets[2100:2550]), 100, regularizer_type='L2')
    nn = load('models.pkl')
    nn.predict(inputs[2550:], targets[2550:])
	
