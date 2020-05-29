#!/usr/bin/python3

import fasttext
import sys


args = sys.argv[1].strip('\n').split('-')
minCount, ngrams, dims, dataset = int(args[0]), int(args[1]), int(args[2]), args[3]

print(minCount, ngrams, dims, dataset)
directory = "/data/s2769670/scriptie/embeddings/"

if dataset == 'non_abusive':
	model = fasttext.train_unsupervised('{}/non_abusive_train.en'.format(dataset), model='skipgram',minCount=minCount, wordNgrams=ngrams, dim=dims)
	model.save_model("{}embeddings_{}_{}_{}_{}.model".format(directory, dataset, minCount, ngrams, dims))
else:
	model = fasttext.train_unsupervised('{}/abusive_train.en'.format(dataset), model='skipgram',minCount=minCount, wordNgrams=ngrams, dim=dims)
	model.save_model("{}embeddings_{}_{}_{}_{}.model".format(directory, dataset, minCount, ngrams, dims))
