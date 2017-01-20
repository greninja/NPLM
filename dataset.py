import itertools
from sklearn.preprocessing import OneHotEncoder
import re
from pprint import pprint
import  numpy as np 
import tensorflow as tf
from nltk.util import ngrams



def OneHotEncoding(text):

	# words = tf.Variable()
	words = re.findall('\w+',text)	

	# One Hot Encoding
	word_vec = [np.zeros(len(words),dtype=int) for _ in xrange(len(words))]

	length = len(words)

	for i in xrange(length):
		word_vec[i][i] = 1

	word_dict = { word : word_vec[i] for word,i in words,len(xrange(word_vec))}
	
	return word_vec


def n_gram(text):


	tokenize = text.word_tokenize(text)
	four_grams = ngrams(tokenize,4)


	


