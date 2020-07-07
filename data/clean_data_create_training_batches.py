#!/usr/bin/python3

import pandas as pd
import numpy as np 
import redditcleaner
import emoji
import re

def filterText(row):
	text = str(row)
	# Filter URLs
	text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "<URL>", text)
	# Filter numbers
	text = re.sub(r'\b\d+\b', "<NUMBER>", text)
	# Filter usernames
	text = re.sub(r'\b@\w\b', "@USER", text)
	# Convert emojis to text
	text = emoji.demojize(text)
	# Clean text of Reddit markup language/characters
	text = redditcleaner.clean(text)
	
	
	
	# Tokenize text
	#tokens = tokenizer.tokenize(text)
	#print(text)
	
	return text


def preprocessing(filterSubreddits):
	# Read in files
	df_abusive = pd.read_csv('reddit/preprocessed_reddit_abusive_large.csv', names=['subreddit', 'old_text', 'labels'], header=0)
	#df_non_abusive = pd.read_csv('reddit/preprocessed_reddit_non_abusive.csv', names=['subreddit', 'text', 'labels'])

	# Preprocess abusive data
	df_abusive['text'] = df_abusive['old_text'].apply(filterText)

	# # Store data for test set in seperate dataframe
	df_abusive_test = df_abusive[df_abusive.subreddit.isin(filterSubreddits)]
	# print(df_abusive_test['subreddit'].value_counts())
	# print(df_abusive_test[['text', 'labels']].head())

	# # Remove testdata subreddit from trainingset
	df_abusive = df_abusive[~df_abusive.subreddit.isin(filterSubreddits)]
	print(df_abusive['subreddit'].value_counts())
	# print(df_abusive[['text', 'labels']].head())

	# Add label to non-abusive data
	#df_non_abusive['labels'] = 'NOT'
	# Preprocess non abusive data 
	#df_non_abusive['text'] = df_non_abusive['old_text'].apply(filterText)

	# Build and return outputfiles
	df_abusive[['subreddit','text', 'labels']].to_csv('reddit/abusive_train_large.csv')
	df_abusive_test[['subreddit','text', 'labels']].to_csv('test/evaluationsets/self/preprocessed_abusive_test.csv')
	#df_non_abusive[['subreddit','text', 'labels']].to_csv('reddit/non_abusive_train.csv')

	



def create_batches(n_batches, n_non_abusive, n_implicit, n_explicit, n_batch_skip=0):
	# Load training data
	df_abusive = pd.read_csv('reddit/abusive_train_large.csv', names=['subreddit', 'text', 'labels'], engine='python')
	df_non_abusive = pd.read_csv('reddit/non_abusive_train.csv', names=['subreddit', 'text', 'labels'], engine='python')
	
	# Shuffle dataset to get variation in abusive data
	df_abusive = df_abusive.sample(frac=1).reset_index(drop=True)
	print("#### Done loading in trainingData")


	df_imp = df_abusive.loc[df_abusive['labels'] == 'IMP']
	df_exp = df_abusive.loc[df_abusive['labels'] == 'EXP']

	for i in range(1 + n_batch_skip, n_batches + n_batch_skip):
		batch_implicit = df_imp.iloc[0 : i * n_implicit]
		batch_explicit = df_exp.iloc[0 : i * n_explicit]
		batch_non = df_non_abusive.iloc[0 : i * n_non_abusive]

		# Combine the data
		batch_abusive = batch_implicit.append(batch_explicit, ignore_index=True)
		batch_combined = batch_abusive.append(batch_non, ignore_index=True)
		# Shuffle the trainingdata
		batch_combined = batch_combined.sample(frac=1).reset_index(drop=True)

		# Filter headers
		batch_combined = batch_combined[batch_combined.labels.isin(['NOT', 'IMP', 'EXP'])]

		# Calculate batchcount
		batch_size = n_non_abusive + n_implicit + n_explicit
		total_documents_current = i * (n_non_abusive + n_implicit + n_explicit)

		# Write to new batchfiles
		batch_combined.to_csv("training/batches/333333/batch_train_{}.csv".format(total_documents_current), sep='\t')


def main():
	# Abusive Subreddits to be filtered for the test data
	filterSubreddits = [
		'niggerspics', 'misogyny', 'niggervideos', 'hitler', 'polacks',
       'niggas', 'pol', 'niggersstories', 'chimpmusic', 'funnyniggers',
       'niggerhistorymonth', 'niglets', 'gibsmedat', 'teenapers',
       'didntdonuffins', 'blackpeoplehate', 'chicongo', 'whitesarecriminals',
       'muhdick', 'apewrangling', 'niggerrebooted', 'beatingfaggots', 'kike',
       'klukluxklan'
		]

	# Preprocess the trainingdata
	#preprocessing(filterSubreddits)

	# Split the data into multiple training batches of 10000
	create_batches(20, 2000, 2000, 2000, 0)




if __name__ == '__main__':
	main()
