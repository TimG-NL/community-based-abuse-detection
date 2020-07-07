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
from keras.models import load_model
from keras.utils import CustomObjectScope




# Attention
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.models import Model
from keras.layers import Concatenate, Conv1D, Activation, TimeDistributed, Flatten, RepeatVector, Permute, multiply
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
import sys
from os import walk


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




def data_preparation(max_len_sent, data_source, distribution, batch_train_file, gold_train_file, classification_type):
	"""
	Load in the training data based on classification_type and data_source
	"""

	# Map explicit and implicit to 0,1 if classification_type == binary
	if classification_type == 'binary':
		mapping_dict = {'NOT': 0, 'IMP' : 1, 'EXP' : 1, '0': 0, '1': 1, '2': 1}
	# Else multi-class classification
	else:
		mapping_dict = {'NOT': 0, 'IMP' : 1, 'EXP' : 2, '0': 0, '1': 1, '2': 2}

	if data_source == "reddit_distant" or data_source == 'reddit+gold':
		# Get training data
		datafile = "../../data/training/batches/{}/batch_train_{}.csv".format(distribution, batch_train_file)
		reddit_df = pd.read_csv(datafile, names=['id', 'subreddit', 'text', 'classes'], sep='\t', header=0, index_col=0, error_bad_lines=False)
		reddit_df.dropna(inplace=True)
		
		reddit_df['labels'] = reddit_df['classes'].apply(lambda x: mapping_dict[str(x)])

	if data_source == "gold_train" or data_source == 'reddit+gold':
		goldtrain_df = pd.read_csv(gold_train_file, names=['id', 'text', 'classes'], sep='\t', header=0, index_col=0, error_bad_lines=False)
		goldtrain_df.dropna(inplace=True)
		
		goldtrain_df['labels'] = goldtrain_df['classes'].apply(lambda x: mapping_dict[str(x)])

	print("##### Loaded in training data #####", file=text_file)
	
	# Decide which data to return
	if data_source == 'reddit+gold':
		# Combine reddit + gold data
		reddit_gold_df = reddit_df.append(goldtrain_df, ignore_index=True).reset_index()
		X = reddit_gold_df['text'].values
		if classification_type == 'multiclass':
			y = to_categorical(reddit_gold_df['labels'].values, num_classes=3)
		else:
			y = reddit_gold_df['labels'].values
	if data_source == 'reddit_distant':
		X = reddit_df['text'].values
		if classification_type == 'multiclass':
			y = to_categorical(reddit_df['labels'].values, num_classes=3)
		else:
			y = reddit_df['labels'].values
	else:
		X = goldtrain_df['text'].values
		if classification_type == 'multiclass':
			y = to_categorical(goldtrain_df['labels'].values, num_classes=3)
		else:
			y = goldtrain_df['labels'].values

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)


	# Create sequences word -> indexes
	tokenizer = Tokenizer(oov_token="_UNK")
	tokenizer.fit_on_texts(x_train)
	xtrain = tokenizer.texts_to_sequences(x_train)
	xtest = tokenizer.texts_to_sequences(x_test)

	# Get vocabulary size 
	vocab_size = len(tokenizer.word_index) + 1

	# Pad and truncate sequences to predefined length
	xtrain = pad_sequences(xtrain, truncating='post', padding='post', maxlen=max_len_sent)
	xtest = pad_sequences(xtest, truncating='post', padding='post', maxlen=max_len_sent) 

	print("######## Done preparing data ###########", file=text_file)
	return xtrain, xtest, y_train, y_test, vocab_size, tokenizer


"""
Load in the pretrained embeddings
"""
def load_fasttext_embeddings(tokenizer, vocab_size):
	embeddings_na = fasttext.load_model("../../data/embeddings/fasttext/embeddings_non_abusive_1_2_300.model")
	embeddings_a = fasttext.load_model("../../data/embeddings/fasttext/embeddings_abusive_1_2_300_large.model")


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
	print('########## Loaded word vectors #############', file=text_file)
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

	print('Loaded %s word vectors.' % len(glove_dict), file=text_file)
	glove_matrix = np.zeros((vocab_size, 300)) #Dimension vector in embeddings

	# Create embedding matrix word_idx: [1, 300]
	for word, index in tokenizer.word_index.items():
		if index > vocab_size - 1:
			break
		else:
			embedding_vector = glove_dict.get(word)
			if embedding_vector is not None:
				glove_matrix[index] = embedding_vector

	print("############ Done creating glove embeddings #############", file=text_file)
	print("{} - {}".format(len(glove_dict.keys()), len(glove_matrix)), file=text_file)
	return glove_dict, glove_matrix
	


"""
Create the LSTM model based on the type of embeddings used
"""

def LSTMmodel(classification_type, experiment_n, vocab_size, maxlen, embedding_source, list_embedding_matrix):
	sentence_input = Input(shape=(maxlen,), dtype='int32')
	if embedding_source == 'glove':
		glove_weights = list_embedding_matrix[0]

		embedding_dim = 300
		embeddings = Embedding(input_dim=vocab_size,
								output_dim=embedding_dim,
								input_length=maxlen,
								weights=[glove_weights],
								trainable=False,
								mask_zero=True)(sentence_input)
		
	elif embedding_source == 'fasttext':
		non_abusive_weights = list_embedding_matrix[0]
		abusive_weights = list_embedding_matrix[1]
		fasttext_weigths = list_embedding_matrix[2]

		# Decide which embeddings to choose
		if experiment_n == 2:
			embedding_dim = 300
			fasttext_weigths = abusive_weights
		else:
			embedding_dim= 600

		print(non_abusive_weights.shape, abusive_weights.shape, fasttext_weigths.shape, file=text_file)
		
		embeddings = Embedding(input_dim=vocab_size,
								output_dim=embedding_dim,
								input_length=maxlen,
								weights=[fasttext_weigths],
								trainable=False,
								mask_zero=True)(sentence_input)

	first_bilstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embeddings)
	#second_bilstm_layer = Bidirectional(LSTM(64, return_sequences=True))(first_bilstm_layer)
	attention_layer = AttentionWithContext()(first_bilstm_layer)
	dense_layer = Dense(64, activation="relu")(attention_layer)
	if classification_type == 'multiclass':
		output_layer = Dense(3, activation="softmax")(dense_layer)
		model = Model(sentence_input,output_layer)
		model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
	elif classification_type == 'binary':
		output_layer = Dense(1, activation="sigmoid")(dense_layer)
		model = Model(sentence_input,output_layer)
		model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

	# print summary of the model
	print(model.summary(), file=text_file)
	print("############ Done building the model #############", file=text_file)

	return model


def train_model(model, x_train, y_train, x_dev, y_dev, filepath):
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint, es]
	model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_dev, y_dev), callbacks=callbacks_list, verbose=1)

	print("############ Done training the model #############", file=text_file)
	return model


def loadTestData(classification_type, tokenizer, max_len_sent):
	test_dict = {}
	test_dict['offenseval2019'] = pd.read_csv("../../data/test/test_offenseval2019.csv", names=['id', 'old_text', 'classes'], sep='\t', header = 0)
	test_dict['abuseval'] = pd.read_csv("../../data/test/test_abuseval.csv", names=['id', 'old_text', 'classes'], sep='\t', header = 0)
	test_dict['offenseval2020'] = pd.read_csv("../../data/test/test_offenseval2020.csv", names=['id', 'old_text', 'classes'], sep='\t', header = 0)
	#test_dict['hateval2019'] = pd.read_csv("../../data/test/test_hateval.csv", names=['id', 'text', 'classes'], sep='\t', header = 0)
	test_dict['reddit_students_self'] = pd.read_csv("../../data/test/test_students_self_reddit.csv", names=['id', 'old_text', 'classes', 'annotator'], sep='\t', header = 0, index_col=0)


	# Map explicit and implicit to 0,1 if classification_type == binary
	if classification_type == 'binary':
		mapping_dict = {'NOT': 0, 'IMP' : 1, 'EXP' : 1, '0': 0, '1': 1, '2': 1}
		for testset in test_dict.keys():
			test_dict[testset]['classes'] = test_dict[testset]['classes'].apply(lambda x: mapping_dict[str(int(x))])
	# Else multi-class classification
	else:
		mapping_dict = {'NOT': 0, 'IMP' : 1, 'EXP' : 2, '0': 0, '1': 1, '2': 2}
		for testset in test_dict.keys():
			test_dict[testset]['classes'] = test_dict[testset]['classes'].apply(lambda x: mapping_dict[str(int(x))])

	print("##### Loaded in test sets #####", file=text_file)
	return test_dict


def evaluation(model, x_test, y_test, classification_type, testset):
	print("######## {}".format(testset), file=text_file)
	loss, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=False)
	print("Test Accuracy: ", acc, file=text_file)

	y_pred = model.predict(x_test)

	if classification_type == 'binary':
		predictions = [1 if prediction > 0.5 else 0 for prediction in y_pred]
	elif classification_type == 'multiclass':
		y_test = np.argmax(y_test, axis=1)
		predictions = np.argmax(y_pred, axis=1)
		if testset in ["offenseval2019", "offenseval2020", "hateval2019"]:
			predictions = [1 if i == 2 else i for i in predictions]
	
	print(confusion_matrix(y_test, predictions), file=text_file)
	print(classification_report(y_test, predictions, digits=4), file=text_file)
	result_dict = classification_report(y_test, predictions, output_dict=True)

	return result_dict




def main(argv):
	# python3 modelLSTM.py classification_type-exp_number-embeddings_source-distribution-batch_size-gold_data
	# python3 modelLSTM.py multiclass-1-fasttext-252550-24000-abuseval
	# python3 modelLSTM.py multiclass-2-glove-NA-NA-abuseval
	# python3 modelLSTM.py multiclass-3-fasttext-NA-NA-offenseval2019
	######### Determine settings of the experiment:
	import tensorflow as tf
	print(tf.config.list_physical_devices('GPU'))



	# binary-1-glove-252550-12000-NA
	arg = argv[1].strip('\n').split('-')

	################ Define experiment settings
	# Main settings:
	classification_type = arg[0]
	experiment_number = int(arg[1])
	embedding_source = arg[2]

	# Assign data sources and distributions to variables
	if experiment_number == 1:
		data_source = "reddit_distant" #gold_train #reddit+gold
		distribution = arg[3]
		batch_train_file = arg[4]
		gold_train_file = "NA"
		model_name = 'lstm-{}-{}-{}-{}-{}-{}.h5'.format(classification_type, experiment_number, embedding_source, data_source, str(distribution), str(batch_train_file))
		tokenizer_name = 'lstm-{}-{}-{}-{}-{}-{}.pickle'.format(classification_type, experiment_number, embedding_source, data_source, str(distribution), str(batch_train_file))
	elif experiment_number == 2:
		data_source = "gold_train"
		distribution = arg[3]
		batch_train_file = arg[4]
		gold_train_file = "../../data/training/gold_train/train_{}.csv".format(arg[5])
		model_name = 'lstm-{}-{}-{}-{}-{}.h5'.format(classification_type, experiment_number, embedding_source, data_source, arg[5])
		tokenizer_name = 'lstm-{}-{}-{}-{}-{}.pickle'.format(classification_type, experiment_number, embedding_source, data_source, arg[5])
	elif experiment_number == 3:
		data_source = "reddit+gold"
		distribution = arg[3]
		batch_train_file = arg[4]
		gold_train_file = "../../data/training/gold_train/train_{}.csv".format(arg[5])
		model_name = 'lstm-{}-{}-{}-{}-{}-{}-{}.h5'.format(classification_type, experiment_number, embedding_source, data_source, str(distribution), str(batch_train_file), arg[5])
		tokenizer_name = 'lstm-{}-{}-{}-{}-{}-{}-{}.pickle'.format(classification_type, experiment_number, embedding_source, data_source, str(distribution), str(batch_train_file), arg[5])
	

	# Define max sentence length
	max_len_sent = 150

	# Write program details to outputfile
	global text_file 
	text_file = open("output/output_lstm-{}-{}-{}-{}-{}-{}.txt".format(classification_type, experiment_number, embedding_source, distribution, batch_train_file, arg[5]), "a+")
	print("############ New model #############", file=text_file)
	print(arg, file=text_file)

	print("Classification_type: {}\n\
		Experiment_number: {}\n\
		Data_source: {}\n\
		Distribution: {}\n\
		Batch_train_file: {}\n\
		Gold_train_file: {}\n\
		Embeddings_source: {}\n\
		".format(classification_type, experiment_number, data_source, distribution, batch_train_file, arg[5], embedding_source), file=text_file)

	# Check whether model has already been trained or not
	existing_models = []
	for (_, _, filenames) in walk('../models_saved/lstm/exp{}/models/'.format(experiment_number)):
		existing_models.extend(filenames)
		break


	# Train the model
	if model_name not in existing_models:
		################ Load in the data
		x_train, x_dev, y_train, y_dev, vocab_size, tokenizer = data_preparation(max_len_sent, data_source, distribution, batch_train_file, gold_train_file, classification_type)
		with open("/data/s2769670/scriptie/models_saved/lstm/exp{}/tokenizers/{}".format(experiment_number, tokenizer_name), 'wb') as handle:
				pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
		print("Training new model!", file=text_file)
	# Load existing model
	else:
		with open("../models_saved/lstm/exp{}/tokenizers/{}".format(experiment_number, tokenizer_name), 'rb') as handle:
			tokenizer = pickle.load(handle)
		vocab_size = len(tokenizer.word_index) + 1
		print("Loading old model!", file=text_file)
	
	if embedding_source == 'glove':
		# load pretrained glove embeddings
		glove_dict, glove_matrix = load_glove_embeddings(tokenizer, vocab_size)
		list_embedding_matrix = [glove_matrix]
	elif embedding_source == 'fasttext':
		# Load in pretrained fasttext embeddings
		non_abusive_matrix, abusive_matrix, combined_matrix = load_fasttext_embeddings(tokenizer, vocab_size)
		list_embedding_matrix = [non_abusive_matrix, abusive_matrix, combined_matrix]


	if model_name not in existing_models:
		################ Build the model
		model = LSTMmodel(classification_type, experiment_number, vocab_size, max_len_sent, embedding_source, list_embedding_matrix)
		################ Train the model
		filepath = "../models_saved/lstm/exp{}/models/{}".format(experiment_number, model_name)
		model = train_model(model, x_train, y_train, x_dev, y_dev, filepath)

		# Load best model
		with CustomObjectScope({'AttentionWithContext': AttentionWithContext}):
			model = load_model("../models_saved/lstm/exp{}/models/{}".format(experiment_number, model_name))
	else:
		with CustomObjectScope({'AttentionWithContext': AttentionWithContext}):
			model = load_model("../models_saved/lstm/exp{}/models/{}".format(experiment_number, model_name))
	################ Evaluation
	#x_test, y_test = load_evaluation_data(tokenizer, max_len_sent)
	#evaluation(model, x_train, y_train, x_test, y_test)
	test_dict = loadTestData(classification_type, tokenizer, max_len_sent)

	# Dictionary for storing the reload_modelsults from the testsets
	results_dict = {}

	# Evaluate model on test sets
	for test_set in test_dict.keys():
		#print("### {} ###".format(testset))
		x_test_seq = tokenizer.texts_to_sequences(test_dict[test_set]['old_text'].astype(str).values)
		x_test_padded = pad_sequences(x_test_seq, truncating='post', padding='post', maxlen=max_len_sent)
		# If multiclass change to one hot labels
		if classification_type == 'binary':
			y_test = test_dict[test_set]['classes'].values
		else:	
			y_test = to_categorical(test_dict[test_set]['classes'].values, num_classes=3)
		
		# for i in zip(test_dict[test_set]['classes'].values, y_test):
		# 	print(i)	

		results = evaluation(model, x_test_padded, y_test, classification_type, test_set)

		# Add results to dictorary {'test_set': result_dict}
		results_dict[test_set] = results



	# Write experiment settings and output to csvfile
	# Open csv file for output program
	print("######## Writing output to csv ########", file=text_file)
	fields=['date','experiment','classification_type', 'embedding_source',
			'data_source', 'distribution','batch_size', 'gold_train_file', 'testdata', 'accuracy_score', 'macro-f1', 
			'label', 'precision', 'recall', 'f1-score', 'support']
	csvfile = open('exp{}_lstm_results.csv'.format(experiment_number), 'a+')
	
	writer = csv.DictWriter(csvfile, fieldnames = fields)    
	writer.writeheader()

	# Add experiment settings to csv file
	for row_counter, testset in enumerate(results_dict.keys()):
		row = {}
		# Write first row
		if row_counter == 0:
			# Get time
			time = datetime.datetime.now()
			date_time = "{}-{}-{}_{}:{}".format(time.day, time.month, time.year, time.hour, time.minute)
			row = {"date": date_time,
				   "experiment": experiment_number,
				   "classification_type": classification_type,
				   "data_source": data_source,
				   "embedding_source": embedding_source,
				   "distribution": distribution,
				   "batch_size": batch_train_file,
				   "gold_train_file": arg[5]
			}
			
		for item_counter, label in enumerate(results_dict[testset].keys()):
			#print(label)
			if item_counter == 0:
				row["testdata"] =  testset
				row["accuracy_score"] = round(results_dict[testset]['accuracy'], 4)
				row["macro-f1"] = round(results_dict[testset]['macro avg']['f1-score'], 4)
				row["label"] = "0"
				row["precision"] = round(results_dict[testset]["0"]['precision'], 4)
				row["recall"] = round(results_dict[testset]["0"]['recall'], 4)
				row["f1-score"] = round(results_dict[testset]["0"]['f1-score'], 4)
				row["support"] = results_dict[testset]["0"]['support']
				#writer.writerow(row)
			else:
				# Write other rows
				if label in ["1", "2", "macro avg", "weighted avg"]:
					row = {"label": label,
						   "precision": round(results_dict[testset][label]['precision'], 4),
						   "recall": round(results_dict[testset][label]['recall'], 4),
						   "f1-score": round(results_dict[testset][label]['f1-score'], 4),
						   "support": results_dict[testset][label]['support']
						   }
				else: 
					continue
			writer.writerow(row)

	writer.writerow({})
	csvfile.close()
	text_file.close()
	


if __name__ == '__main__':
	main(sys.argv)