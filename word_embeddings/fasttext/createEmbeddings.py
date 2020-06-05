#!/usr/bin/python3

import fasttext
import sys


args = sys.argv[1].strip('\n').split('-')
minCount, ngrams, dims, dataset = int(args[0]), int(args[1]), int(args[2]), args[3]

print(minCount, ngrams, dims, dataset)
directory = "../../data/embeddings/fasttext/input"

if dataset == 'non_abusive':
	model = fasttext.train_unsupervised('{}/non_abusive_train_fasttext.en'.format(directory), model='skipgram',minCount=minCount, wordNgrams=ngrams, dim=dims)
	model.save_model("{}embeddings_{}_{}_{}_{}.model".format(directory, dataset, minCount, ngrams, dims))
else:
	model = fasttext.train_unsupervised('{}/abusive_train_fasttext_large.en'.format(directory), model='skipgram',minCount=minCount, wordNgrams=ngrams, dim=dims)
	model.save_model("{}/embeddings_{}_{}_{}_{}_large.model".format(directory, dataset, minCount, ngrams, dims))
