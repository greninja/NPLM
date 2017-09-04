import itertools
import  numpy as np 
from nltk.util import ngrams
from collections import OrderedDict 

#from sklearn.preprocessing import OneHotEncoder
#import re
#import tensorflow as tf

def OneHotEncoding(ordered_dict):
	OneHotEncoding_vector = [np.zeros(len(ordered_dict),dtype=int) \
						 		for _ in xrange(len(ordered_dict))]

	# A mapping of words and its vector representation
	WordVecDict = OrderedDict()
	for key,value in ordered_dict.iteritems():
		OneHotEncoding_vector[value][value] = 1	
		WordVecDict[key] = OneHotEncoding_vector[value]
	return WordVecDict

def n_gram(text):
	tokenize = text.word_tokenize(text)
	four_grams = ngrams(tokenize,4)


	


