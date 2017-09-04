import codecs
import os
import re
import numpy as np 

ROOT =  os.path.abspath(os.path.dirname(__file__))

class load_data:
	
	def __init__(self, data_dir, batch_size, epoch, embedding_size):

		self.data_dir = os.path.join(ROOT, "data", "lalaland")
		self.batch_size = batch_size
		self.epoch = epoch
		self.embedding_size = embedding_size
		
		with open(self.data_dir,"r") as f:
			text = f.read()
		regex = re.compile(r'\([^)]*\)') # Removing words inside brackets
		sub_text = regex.sub('', text)
		self.words = re.findall('\w+', sub_text)
		self.vocab = set(words)
		
	def generate_word_embeddings(self):
		mapping = { word : np.random.rand(self.embedding_size) for word in self.vocab }
		embedding  = [ mapping[word] for word in self.words]
		return embedding
