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
	"""
	def __init__(self,epoch,hidden_dim,batch_size,learning_rate,window):
		self.epoch = epoch
		#self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.learning_rate = learning_rate

		self.embeddings = get_embeddings()
		shape = embeddings.shape
		self.vocablen = len(embeddings)
		self.cardinality = reduce(lambda x,y : x * y,shape) # cardinality is the total number of elements in embeddings matrix. \
												  		# Weight is assigned to each single element.
		self.window = window
		self.j = 0
		with tf.name_scope('projection_to_hidden'):
			self.c = tf.Variable(embeddings.reshape(cardinality,1),name='c') # c is the embedding matrix converted into column matrix to supplement matrix multiplications
			self.M = tf.Variable(tf.random_normal([hidden_dim,cardinality],dtype=tf.float32),name='M') #  M is the weight matrix between the projection layer and hidden layer
			self.d = tf.Variable(tf.zeros([hidden_dim,1)) # Hidden layer biases
			

		with tf.name_scope('hidden_to_output'):		
			self.b = tf.Variable(tf.zeros([vocablen])) # Output layer biases 
			self.U = tf.Variable(tf.random_normal([vocablen,hidden_dim],dtype=float32),name='U') # Weight matrix hidden-to-output layer
			self.W = tf.Variable(np.random.rand(vocablen,cardinality),name='W') # Weight matrix word-features to output.
	"""
	def __init__(self,epoch,hidden_dim,batch_size,learning_rate,window):
		self.epoch = epoch
		#self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.learning_rate = learning_rate

		self.embeddings = get_embeddings()
		shape = embeddings.shape
		self.vocablen = len(embeddings)
		self.cardinality = reduce(lambda x,y : x * y,shape) # cardinality is the total number of elements in embeddings matrix. \
												  		# Weight is assigned to each single element.
		self.window = window
		self.j = 0

		self.c = embeddings.reshape(self.cardinality,1)
		self.M = np.random.rand(hidden_dim,self.cardinality)
		self.d = np.zeros((hidden_dim,1))
			
		self.b = np.zeros((vocablen,1)) # Output layer biases 
		self.U = np.random.rand(vocablen,hidden_dim) # Weight matrix Hidden to output
		#self.W = tf.Variable(np.random.rand(vocablen,cardinality),name='W') # Weight matrix word-features to output.
		self.yP = [] # List of output probabilities

	def forward_phase():
		Hidden = tf.tanh( tf.matmul(self.M, self.c) + self.d )
		y = tf.nn.softmax( tf.matmul(self.U,Hidden) + self.b )
		loss = -(tf.log(y))
		optimizer = tf.train.GradientDescentOptimizer(1.0)
		train_op = optimizer.minimize(loss)
		return loss, train_op

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
		return 	np.exp(y) / sum(np.exp(y))

	def backward_phase():

		# Compute the gradients here
	
	def train():
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(self.epoch):
				for _ in range(len(embeddings) // self.batch_size):

					# Train the net here.		
	
	#def train_manually():
	#ravelled = r 
	#for _ in xrange(h-1):
	#	ravelled = np.vstack((ravelled,r)) # ravelled matrix is the 
	
	def cross_entropy(true,predicted):
		return sum(map(lambda t,x : t * np.log(x) + (1-t) * np.log(1-x)))

