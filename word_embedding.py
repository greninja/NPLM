
import re
import argparse
import collections
import random
import numpy as np
import tensorflow as tf
import math
# Import One hot encoder here

vocabulary_size = 400
FILE_PATH = '/home/shadab/python/testing/t1'

def data_generate(filename):

	with open(filename,'r') as f:
		text = f.read()
	regex = re.compile(r'\([^)]*\)') 
	sub_text = regex.sub('',text)
	data = re.findall('\w+',sub_text)
	return data

def wordDictionary(filename):
	# Generate the data
	words = data_generate(filename)

	word_dict = collections.OrderedDict()
	
	# Index for new dictionary
	index = 0

	for word in words:
		if word not in word_dictionary:
			word_dict[word] = index
			index += 1

	return word_dict

def build_dataset(words):
	
	count = [['rare', -1]]
   
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	
	for word, _ in count:
		dictionary[word] = len(dictionary)
	
	data = list()
	rare_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  
			rare_count = rare_count + 1
		data.append(index)
	count[0][1] = rare_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

def generate_batch(batch_size,skip_window,num_skips):
	global data_index
	assert batch_size % num_skips == 0
	batch = np.ndarray(shape=(batch_size), dtype = np.int32)
	labels = np.ndarray(shape=(batch_size,1), dtype = np.int32)

	# Span includes the middle element and the surrounding elements. Hence plus one.
	span = 2 * skip_window + 1

	buffer = collections.deque(maxlen=span)

	for _ in xrange(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)

	for i in range(batch_size//num_skips):
		target = skip_window # Middle element is the target
		targets_to_avoid = [ skip_window ]

		for j in range(num_skips):
			while target in targets_to_avoid: # we dont want already included word
				target = random.randint(0,span-1) # to select a random word from the buffer
			targets_to_avoid.append(target) 

			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j] = buffer[target] # here labels are the surrounding words
		
		# This is to get the next element in the queue.
		buffer.append(data[data_index]) 
		data_index = (data_index+1) % len(data)

	return batch, labels

"""
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)


print ('data :', [reverse_dictionary[d] for d in data[:8]])

for i in range(8):
	print(batch[i], reverse_dictionary[batch[i]], \
			'->', labels[i, 0], reverse_dictionary[labels[i, 0]])

"""

def skip_gram():
	
	
	batch_size = args.batch_size
	embedding_size = args.embedding_size # Feature vector size
	skip_window = args.skip_window
	num_skips = args.num_skips

	# Negative sampling

	valid_size = 16
	valid_window = 100
	valid_examples = np.random.choice(valid_window,valid_size,replace=False)
	num_sampled = 64 # Number of negative samples to sample.

	# Creating the computation graph

	graph = tf.Graph()

	with graph.as_default():


		# Inputs
		train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
		train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
		valid_dataset = tf.constant(valid_examples,dtype=tf.int32)

		with tf.device('/cpu:0'):

			embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size], \
													-1.0,1.0))

			embed = tf.nn.embedding_lookup(embeddings,train_inputs)


			# Construct the variables for the NCE loss
			nce_weights = tf.Variable(
				tf.truncated_normal([vocabulary_size, embedding_size],
								stddev=1.0 / math.sqrt(embedding_size)))
			nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

		# Computing NCE loss
		loss = tf.reduce_mean(
		tf.nn.nce_loss(weights=nce_weights,
				 biases=nce_biases,
				 labels=train_labels,
				 inputs=embed,
				 num_sampled=num_sampled,
				 num_classes=vocabulary_size))

		# Construct the SGD optimizer using a learning rate of 1.0.
		optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

		# Compute the cosine similarity between minibatch valid_examples and all embeddings.
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(
		normalized_embeddings, valid_dataset)
		similarity = tf.matmul(
		valid_embeddings, normalized_embeddings, transpose_b=True)

		# Add variable initializer.
		init = tf.global_variables_initializer()

		# Step 5: Begin training.
		num_steps = 10000

		with tf.Session(graph=graph) as session:
		# We must initialize all variables before we use them.
			init.run()
			print("Initialized")

			average_loss = 0
			for step in xrange(num_steps):
				batch_inputs, batch_labels = generate_batch(
					batch_size, num_skips, skip_window)
				feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

				# We perform one update step by evaluating the optimizer op (including it
				# in the list of returned values for session.run()
				_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
				average_loss += loss_val

				if step % 2000 == 0:
				  if step > 0:
					average_loss /= 2000
				  # The average loss is an estimate of the loss over the last 2000 batches.
				  print("Average loss at step ", step, ": ", average_loss)
				  average_loss = 0

		
			final_embeddings = normalized_embeddings.eval(session=session)
	
	return final_embeddings


if __name__=="__main__":

	parser = argparse.ArgumentParser(description="Parameters for skip gram model")

	parser.add_argument('-b','--batch_size',action='store',type=int, default=128,\
						help='batch_size for training the model')
	parser.add_argument('-e','--embedding_size',action='store',type=int, default=128,\
						help = 'Number of the dimensions of the feature vector')
	parser.add_argument('-s','--skip_window',action='store',type=int, default=1, \
						help = 'Window size of the words around the context word')
	parser.add_argument('-ns','--num_skips',action='store',type=int, default=2,\
						help = 'Number of times an input can be used in each batch')
		
	args = parser.parse_args()
	
	words = data_generate(FILE_PATH)
	data,count,dictionary,reverse_dictionary = build_dataset(words)
	data_index = 0

	word_embeddings = skip_gram()
	








