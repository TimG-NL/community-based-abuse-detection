#!/usr/bin/python3

import pandas as pd
import csv
import re
import emoji
import redditcleaner
from nltk.tokenize import TweetTokenizer




def filterText(text, tokenizer):
	# Filter URLs
	text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "<URL>", text)
	# Filter numbers
	text = re.sub(r'\b\d+\b', "<NUMBER>", text)
	# Filter usernames
	text = re.sub(r'\b@\w\b', "@USER", text)
	
	# Convert emojis to text
	text = emoji.demojize(text)

	text = redditcleaner.clean(text)

	# Tokenize text
	tokens = tokenizer.tokenize(text)


	
	return " ".join(tokens)

def createFasttextEmbeddingInput(dataset):
	"""
	Create file for training the fasttext embeddings
	"""
	readfile = "../data/reddit/preprocessed_reddit_{}_large.csv".format(dataset)
	df = pd.read_csv(readfile, header=0, engine='python')
	outputfile = "{}_train_fasttext_large.en".format(dataset)
	with open(outputfile, "a+", encoding='utf-8') as f:
	    comments = df.iloc[:, 1].values
	    for comment in comments:
	        f.write(str(comment) + "\n")


def preprocessComments(dataset):
	tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

	# Non-abusive locations
	years = ['2012', '2013', '2014','2015', '2016', '2017']
	months = ['01', '04', '07', '10']
		
	# Choose whole text, 1 sentence or 2 sentences
	files = ['reddish_', 'reddish1sent_', 'reddish2sent_']



	if dataset == "non_abusive":
		csvfile = "../data/reddit/preprocessed_reddit_non_abusive.csv"
		fieldnames = ['subreddit', 'text', 'labels']
		with open(csvfile, "a+", encoding='utf-8') as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for year in years:
				for month in months:
					print(year, month)
					file = "../data/reddit/non-abusive/{}/{}{}-{}.csv".format(year, files[0], year, month)
					df = pd.read_csv(file, header=None)
					
					# Drop empty rows
					df.dropna(subset=[9], inplace=True)

					# Assign label to non-abusive data
					df['labels'] = "NOT"

					# Clean message and add (subreddit, text) to csvfile
					rows = df.iloc[:, [4,9, 11]].values
					for row in rows:
						clean_comment = filterText(row[1], tokenizer)
						row_dict = {'subreddit':row[0] ,'text': clean_comment, 'labels': row[2]}
						writer.writerow(row_dict)
	
	elif dataset == "abusive":
		input_file = "../data/reddit/abusive/reddish.csv"
		
		csvfile = "../data/reddit/preprocessed_reddit_abusive_large.csv"
		fieldnames = ['subreddit', 'text', 'labels']

		with open(csvfile, "a+", encoding='utf-8') as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()

			# read inputfile
			df = pd.read_csv(input_file, header=None)

			# Drop empty rows
			df.dropna(subset=[9], inplace=True)

			# Clean messages and add (message, labels) to csvfile
			rows = df.iloc[:, [4,9,10]].values
			for row in rows:
				clean_comment = filterText(row[1], tokenizer)
				row_dict = {'subreddit': row[0],'text': clean_comment, 'labels': row[2]}
				writer.writerow(row_dict)
				

def main():
	dataset = "abusive"
	#preprocessComments(dataset)
	createFasttextEmbeddingInput(dataset)


if __name__ == '__main__':
	main()