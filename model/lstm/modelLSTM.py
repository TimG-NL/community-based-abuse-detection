#!/usr/bin/python3

# Preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# Model Building
from keras.models import Sequential, Model
from keras.layers import Bidirectional, LSTM, Dense, Embedding, Dropout, Input, concatenate

from keras.callbacks import ModelCheckpoint, EarlyStopping




# Attention
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.models import Model
from keras.layers import Concatenate, Conv1D, Activation, TimeDistributed, Flatten, RepeatVector, Permute,multiply
from keras.layers import GlobalMaxPool1D, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, SpatialDropout1D, BatchNormalization


# Evaluation
from sklearn.metrics import confusion_matrix, classification_report


# General imports
import pandas as pd
import fasttext
import numpy as np 
import pickle
import random
import csv
import datetime


class AttentionWithContext(Layer):
	'''
		Attention operation, with a context/query vector, for temporal data.
		Supports Masking.
		Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
		'Hierarchical Attention Networks for Document Classification'
		by using a context vector to assist the attention
		# Input shape
			3D tensor with shape: `(samples, steps, features)`.
		# Output shape
			2D tensor with shape: `(samples, features)`.
		:param kwargs:
		Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
		The dimensions are inferred based on the output shape of the RNN.
		Example:
			model.add(LSTM(64, return_sequences=True))
			model.add(AttentionWithContext())
		'''

	def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):
		self.supports_masking = True
		self.init = initializers.get(init)
		self.kernel_initializer = initializers.get('glorot_uniform')

		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)

		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)

		super(AttentionWithContext, self).__init__(**kwargs)

	def build(self, input_shape):
		print(self.name)
		self.kernel = self.add_weight(shape=(input_shape[-1], 1),
								 initializer=self.kernel_initializer,
								 name='{}_W'.format(self.name),
								 regularizer=self.kernel_regularizer,
								 constraint=self.kernel_constraint)
		self.b = self.add_weight(shape=(input_shape[1],),
								 initializer='zero',
								 name='{}_b'.format(self.name),
								 regularizer=self.bias_regularizer,
								 constraint=self.bias_constraint)

		self.u = self.add_weight(shape=(input_shape[1],),
								 initializer=self.kernel_initializer,
								 name='{}_u'.format(self.name),
								 regularizer=self.kernel_regularizer,
								 constraint=self.kernel_constraint)
		self.built = True

	def compute_mask(self, input, mask):
		return None

	def call(self, x, mask=None):
		multData =  K.dot(x, self.kernel)
		multData = K.squeeze(multData, -1)
		multData = multData + self.b

		multData = K.tanh(multData)

		multData = multData * self.u
		multData = K.exp(multData)

		# apply mask after the exp. will be re-normalized next
		if mask is not None:
			mask = K.cast(mask, K.floatx())
			multData = mask*multData

		# in some cases especially in the early stages of training the sum may be almost zero
		# and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
		# a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
		multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		multData = K.expand_dims(multData)
		weighted_input = x * multData
		return K.sum(weighted_input, axis=1)


	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1],)




def data_preparation(max_len_sent, data_source):
	# Get training data
	datafile = "../../data/training/batches/6000/batch_train_36000.csv"
	train_df = pd.read_csv(datafile, names=['id', 'text', 'classes'], header = 0, index_col=0, error_bad_lines=False, sep="\t")
	train_df.dropna(inplace=True)
	mapping_dict = {'NOT': 0, 'IMP' : 1, 'EXP' : 2}
	train_df['labels'] = train_df['classes'].apply(lambda x: mapping_dict[x])



	x = train_df['text'].values	
	y = train_df['labels'].values
	y_onehot = to_categorical(y, num_classes=3)

	x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.1, random_state=123)


	# Create sequences word -> indexes
	tokenizer = Tokenizer(oov_token='<UNK>')
	tokenizer.fit_on_texts(x)
	xtrain = tokenizer.texts_to_sequences(x_train)
	xtest = tokenizer.texts_to_sequences(x_test)

	# Get vocabulary size 
	vocab_size = len(tokenizer.word_index) + 1

	# Pad and truncate sequences to predefined length
	xtrain = pad_sequences(xtrain, truncating='post', padding='post', maxlen=max_len_sent)
	xtest = pad_sequences(xtest, truncating='post', padding='post', maxlen=max_len_sent) 


	with open('tokenizer.pickle', 'wb') as handle:
			pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


	print("######## Done preparing data ###########")
	return xtrain, xtest, y_train, y_test, vocab_size, tokenizer


"""
Load in the pretrained embeddings
"""
def load_fasttext_embeddings(tokenizer, vocab_size):
	embeddings_na = fasttext.load_model("../../data/embeddings/fasttext/embeddings_non_abusive_1_2_300.model")
	embeddings_a = fasttext.load_model("../../data/embeddings/fasttext/embeddings_abusive_1_2_300.model")


	non_abusive_matrix = glove_matrix = np.zeros((vocab_size, 300)) #Dimension vector in embeddings
	abusive_matrix = np.zeros((vocab_size, 300)) #Dimension vector in embeddings
	combined_matrix = np.zeros((vocab_size, 600)) # Abusive + non_abusive embeddings

	for word, index in tokenizer.word_index.items():
		if index > vocab_size - 1:
			break
		else:
			embedding_vector_na = embeddings_na.get_word_vector(word)
			embedding_vector_a = embeddings_a.get_word_vector(word)
			if embedding_vector_na is not None:
				non_abusive_matrix[index] = embedding_vector_na
			if embedding_vector_a is not None:
				abusive_matrix[index] =  embedding_vector_a

			# Make matrix with 600 dimensions
			if embedding_vector_a is not None and embedding_vector_na is not None:
				combined_matrix[index] =  np.hstack((embedding_vector_na, embedding_vector_a))
	print('########## Loaded word vectors #############')
	return non_abusive_matrix, abusive_matrix, combined_matrix


def load_glove_embeddings(tokenizer, vocab_size):
	glove_dict = dict()
	with open('../../data/embeddings/glove/glove.840B.300d.txt', 'r', encoding='utf8' ) as f:
		for line in f:
			values = line.split(' ')
			try:
				word = values[:-300][0]
				vector = np.asarray(values[-300:], 'float32')
			except:
				continue
			glove_dict[word] = vector

	print('Loaded %s word vectors.' % len(glove_dict))
	glove_matrix = np.zeros((vocab_size, 300)) #Dimension vector in embeddings

	# Create embedding matrix word_idx: [1, 300]
	for word, index in tokenizer.word_index.items():
		if index > vocab_size - 1:
			break
		else:
			embedding_vector = glove_dict.get(word)
			if embedding_vector is not None:
				glove_matrix[index] = embedding_vector

	print("############ Done creating glove embeddings #############")
	print("{} - {}".format(len(glove_dict.keys()), len(glove_matrix)))
	return glove_dict, glove_matrix
	


"""
Create the LSTM model based on the type of embeddings used
"""

def LSTMmodel(vocab_size, maxlen, embedding_source, list_embedding_matrix):
	sentence_input = Input(shape=(maxlen,), dtype='int32')
	if embedding_source == 'glove':
		glove_weights = list_embedding_matrix[0]

		embedding_dim = 300
		embeddings = Embedding(input_dim=vocab_size,
								output_dim=embedding_dim,
								input_length=maxlen,
								weights=[glove_weights],
								trainable=False)(sentence_input)
		
	elif embedding_source == 'fasttext':
		non_abusive_weights = list_embedding_matrix[0]
		abusive_weights = list_embedding_matrix[1]
		combined_weigths = list_embedding_matrix[2]

		print(non_abusive_weights.shape, abusive_weights.shape, combined_weigths.shape)
		
		embedding_dim = 600
		embeddings = Embedding(input_dim=vocab_size,
								output_dim=embedding_dim,
								input_length=maxlen,
								weights=[combined_weigths],
								trainable=False)(sentence_input)

	model = Bidirectional(LSTM(100, return_sequences=True,dropout=0.50),merge_mode='concat')(embeddings)
	model = TimeDistributed(Dense(100,activation='relu'))(model)
	model = Flatten()(model)
	model = Dense(100,activation='relu')(model)
	output_layer = Dense(3,activation='softmax')(model)
	
	model = Model(sentence_input,output_layer)
	model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])


	# sentence_input = Input(shape=(maxlen,), dtype='int32')

	# if embedding_source == 'glove':
	# 	glove_weights = list_embedding_matrix[0]

	# 	embedding_dim = 300
	# 	glove_emb = Embedding(input_dim=vocab_size,
	# 							output_dim=embedding_dim,
	# 							input_length=maxlen,
	# 							weights=[glove_weights],
	# 							mask_zero=True,
	# 							trainable=False)(sentence_input)

	# elif embedding_source == 'fasttext':
	# 	non_abusive_weights = list_embedding_matrix[0]
	# 	abusive_weights = list_embedding_matrix[1]
	# 	combined_weigths = list_embedding_matrix[2]

		
	# 	print(non_abusive_weights.shape, abusive_weights.shape, combined_weigths.shape)
		
	# 	embedding_dim = 600
	# 	fasttext_emb = Embedding(input_dim=vocab_size,
	# 							output_dim=embedding_dim,
	# 							input_length=maxlen,
	# 							weights=[combined_weigths],
	# 							mask_zero=True,
	# 							trainable=False)(sentence_input)

	
	# biLSTM_layer = Bidirectional(LSTM(units=128, activation="relu", dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(glove_emb)
	# #dropout_layer = Dropout(0.3)(biLSTM_layer)
	# attention_layer = AttentionWithContext()(biLSTM_layer)
	# output_layer = Dense(3, activation="softmax")(attention_layer)

	# model = Model(sentence_input, output_layer)
	# model.compile(optimizer="adam", loss="categorical_crossentropy", 
	# 	 metrics=['accuracy'])

	# print summary of the model
	print(model.summary())
	print("############ Done building the model #############")

	return model


def train_model(model, x_train, y_train, x_dev, y_dev):
	filepath = "model.h5"
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint, es]
	model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_dev, y_dev), callbacks=callbacks_list, verbose=1)

	print("############ Done training the model #############")
	return model


def load_evaluation_data(tokenizer, max_len_sent):
	# Retrieve test data
	datafile = "../../data/test/test_abuseval.csv"
	test_df = pd.read_csv(datafile, names=['id', 'text', 'classes'], header = 0, index_col=0, error_bad_lines=False, sep="\t")

	x_test = test_df['text'].values	

	x_test = tokenizer.texts_to_sequences(x_test)

	x_test = pad_sequences(x_test, truncating='post', padding='post', maxlen=max_len_sent)



	y = test_df['classes'].values
	y_onehot = to_categorical(y, num_classes=3)

	return x_test, y_onehot


def evaluation(model, x_train, y_train, x_test, y_test):
	loss, acc = model.evaluate(x_train, y_train, batch_size=32, verbose=False)
	print("Training Accuracy: ", acc)
	loss, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=False)
	print("Test Accuracy: ", acc)

	y_pred = model.predict(x_test)
	y_pred = np.argmax(y_pred, axis=1)
	print(y_pred)

	y_test = np.argmax(y_test, axis=1)
	print(y_test)

	cm = confusion_matrix(y_test, y_pred)
	report = classification_report(y_test, y_pred)
	print(cm)
	print(report)

	# result = zip(x_test, y_test, y_pred)
	# for i in result:
	# 	print(i)


def main():
	################ Define experiment settings
	n_labels = 3 # train the model binary or multiclass
	data_source = "reddit_distant" #gold_train #reddit+gold



	# Define max sentence length
	max_len_sent = 150


	################ Load in the data
	x_train, x_dev, y_train, y_dev, vocab_size, tokenizer = data_preparation(max_len_sent, data_source)
	
	
	
	embedding_source = 'glove'
	if embedding_source == 'glove':
		# load pretrained glove embeddings
		glove_dict, glove_matrix = load_glove_embeddings(tokenizer, vocab_size)
		list_embedding_matrix = [glove_matrix]
	elif embedding_source == 'fasttext':
		# Load in pretrained fasttext embeddings
		non_abusive_matrix, abusive_matrix, combined_matrix = load_fasttext_embeddings(tokenizer, vocab_size)
		list_embedding_matrix = [non_abusive_matrix, abusive_matrix, combined_matrix]


	################ Build the model
	model = LSTMmodel(vocab_size, max_len_sent, embedding_source, list_embedding_matrix)

	################ Train the model
	model = train_model(model, x_train, y_train, x_dev, y_dev)
	
	################ Evaluation
	x_test, y_test = load_evaluation_data(tokenizer, max_len_sent)
	evaluation(model, x_train, y_train, x_test, y_test)



if __name__ == '__main__':
	main()