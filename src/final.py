"""
This is an implementation of a feedforward neural network for
language modelling (prediction of the next word given on previous
ones) based on a given corpus. Keeping the number of hidden layers
one (with no skip connections). Activation on hidden layer is the 
nonlinear tanh function and on the output is softmax

Reference : A Neural probabilistic language model, Bengio et al.
"""

import re
import argparse
import numpy as np 
import tensorflow as tf 


def xavier_init(n_inputs, n_outputs, uniform=True): # Not using it anymore
  """
  Used to initialize weights using xavier_initialization method.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  
  Reference : Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  
  Parameters
  ----------
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  	"""
  if uniform:
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)

def get_batch(X, size):
	"""
	Used to generate batches for training
	"""
	a = np.random.choice(len(X), size, replace=False)
	return X[a]


def data_generate(filename):
	"""
	Generate a list of words from the given filename (maintaining the order of words).
	This list acts as the vocabulary which is feeded as input to the neural network.

	Parameters 
	----------
	filename : input text file (might consist of stop words)

	""" 
	with open(filename,'r') as f:
		text = f.read()
	regex = re.compile(r'\([^)]*\)')  
	sub_text = regex.sub('',text)
	data = re.findall('\w+',sub_text)
	return data

class nnlm:
	"""
	Feedforward neural network architecture for language modelling

	Parameters
	----------
	epoch : number of epochs for training. One epoch is equal to one forward and \
				backward of for all training examples
	hidden_dim : Number of hidden nodes
	learning_rate : learning rate of the network. Typical value : 0.01
	vocablen : total number of words in the input text file 
	window : size of past context of each next word to be predicted 
	feature_size : size of each word feature vector
	"""
	def __init__(self, epoch, hidden_dim, learning_rate, vocablen, window, feature_size):
		self.epoch = epoch
		self.hidden_dim = hidden_dim
		self.learning_rate = learning_rate
		self.vocablen = vocablen 
		self.window = window
		self.feature_size = feature_size
		

	def initialize_network(self):
		self.wordFeatureMatrix = tf.Variable(tf.random_normal([self.vocablen, \
												self.feature_size]))
		self.hiddenLayerWeights = tf.Variable(tf.random_normal([self.hidden_dim, \
												self.window * self.feature_size]))
		self.outputLayerWeights = tf.Variable(tf.random_normal([self.vocablen, \
												self.hidden_dim]))
		self.outputBiases = tf.Variable(tf.zeros([self.vocablen, 1]))
		self.hiddenBiases = tf.Variable(tf.zeros([self.hidden_dim, 1]))


	def forward_phase(self, batch):
		hidden = tf.tanh(tf.matmul(self.hiddenLayerWeights, batch) + self.hiddenBiases)
		yhat = tf.nn.softmax(tf.matmul(self.outputLayerWeights, hidden) + self.outputBiases)
		return yhat

	def forward_phase_manually():
		
		for i in xrange(vocablen-self.window+1):
			
			context = self.embeddings[i:i+self.window]
			context_cardinality = context.shape[0] * context.shape[1]
			sliced_c = context.reshape(context_cardinality,1)
			sliced_M = self.M[:,self.j:self.j+context_cardinality]
			self.j += context_cardinality
			Hidden = np.dot(sliced_M,sliced_c) + self.d # Hidden layer neurons' weighted sum

			#Activation of hidden layer neurons
			a = np.tanh(Hidden)

			sliced_U = self.U[i:i+self.window]
			sliced_b = b[i:i+self.window]
			k = 0
			y = []
			for k in xrange(self.window):
				yk = sliced_b[k] + np.matmul(sliced_U[k],a)
				y.append(yk)

			y = self.softmax(y)
			yP.append(y[-1])

	def softmax(self, y):
		return np.exp(y) / sum(np.exp(y))

	def train():
		initialize_network()	
		
		batch = tf.placeholder(tf.float16, shape=(self.window*self.feature_size, 1))

		y = forward_phase(batch)
		loss = -(tf.log(y))
		optimizer = tf.train.GradientDescentOptimizer(1.0)
		train_op = optimizer.minimize(loss)		
		

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			total_loss = 0
			for epoch in range(self.epoch):
				for i in range(0, self.vocablen-self.window):
					concFeatureVec = self.wordFeatureMatrix[i,i+self.window].ravel() 
					_, c = sess.run([train_op, loss], feed_dict={batch : concFeatureVec})
				total_loss += c
				print "For epoch {0}, the loss is {1}".format(epoch, total_loss)

		print "Training ends!"
	
def main():

	parser = argparse.ArgumentParser(description="Parameters for neural network")

	parser.add_argument('-e','--epoch', action='store', type=int, default=10,\
						help='number of epochs to run NN on')
	parser.add_argument('-h','--hidden_dim',action='store',type=int, default=20,\
						help='number of hidden layer nodes')
	parser.add_argument('-l','--learning_rate',action='store',type=int, default=0.01,\
						help='learning rate of the neural network')
	parser.add_argument('-w','--window',action='store',type=int, default=5,\
						help='size of the context')
	parser.add_argument('-f','--feature_size',action='store',type=int, default=100,\
						help='dimension of each feature vector')
	parser.add_argument('filename', action='store', \
						help='filename or filepath (if in another directory')
	args = parser.parse_args()
	words = data_generate(args.filename)
	vocablen = len(words)
	nn_object = nnlm(args.epoch, args.hidden_dim, args.learning_rate, \
					vocablen, args.window, args.feature_size)

	nn_object.train()


if __name__=="__main__":
	main()
