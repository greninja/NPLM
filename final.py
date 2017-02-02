import tensorflow as tf 
import numpy as np 
from word_embedding import main

def get_embeddings():
	embeddings = main()
	return embeddings
	#print embeddings.shape()

# h = 100 # Hidden layer units
# m = 128 # No. of features associated with each word 
# vocablen = len(embeddings)
 # biases
 # hidden layer biases
 def get_batch(X, size):
     a = np.random.choice(len(X), size, replace=False)
 	 return X[a]


class FeedForward:
	
	def __init__(self,epoch,hidden_dim,batch_size,learning_rate):
		self.epoch = epoch
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.learning_rate = learning_rate

		embeddings = get_embeddings()
		shape = embeddings.shape
		vocablen = len(embeddings)
		cardinality = reduce(lambda x,y : x * y,shape) # cardinality is the total number of elements in embeddings matrix. \
												  		# Weight is assigned to each single element.
		c = tf.Variable(embeddings.reshape(cardinality,1),name='c') # c is the embedding matrix converted into column matrix to supplement matrix multiplications
		M = tf.Variable(np.random.rand(self.hidden_dim,cardinality),name='M') #  M is the weight matrix between the projection layer and hidden layer
		b = "Output layer biases"
		d = "Hidden layer biases with h units"
		U = tf.Variable(np.random.rand(vocablen,h),name='U') # Weight matrix hidden-to-output layer
		W = tf.Variable(np.random.rand(vocablen,cardinality),name='W') # Weight matrix word-features to output.

		

	#def feedforward():
	def train():

		Hidden = np.dot(M,c) # Hidden layer neurons


	#ravelled = r 
	#for _ in xrange(h-1):
	#	ravelled = np.vstack((ravelled,r)) # ravelled matrix is the 

