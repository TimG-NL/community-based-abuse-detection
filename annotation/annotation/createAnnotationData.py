#!/usr/bin/python3
import pandas as pd
from random import shuffle
from sklearn.utils import shuffle
import numpy as np

def createAnnotationCSV():
	"""
	Creates annotation.csv where the data from 2016-02, 2017-05 and 2017-08 are 
	concatenated and shuffled
	"""
	years =  ["2016-02", "2017-05", "2017-08"]
	fieldnames = ["year", "month", 'comment_id', 'parent_text', 'subreddit', "timestamp", "gilded", "upvotes", "downvotes", "text", "abusive"]
	df2016_02 = pd.read_csv("output/reddish_2016-02.csv", names=fieldnames)
	df2017_05 = pd.read_csv("output/reddish_2017-05.csv", names=fieldnames)
	df2017_08 = pd.read_csv("output/reddish_2017-08.csv", names=fieldnames)

	print(df2016_02.shape)
	df = pd.concat([df2016_02, df2017_05, df2017_08], axis=0)

	print(df.shape)
	print(df.columns)


	df = shuffle(df)
	
	df['text'].replace('', np.nan, inplace=True)
	df.dropna(subset=["text"], inplace=True)
	df.reset_index(inplace=True, drop=True)
	df2 = df[df['text'].notnull()].astype(str)
	print(df2.shape)

	df2.iloc[1:10000, :].to_csv("annotation_clean.csv", columns=fieldnames)



	df = pd.read_csv("annotation_clean.csv", header=0, engine='python')
	print(df.head())
	outputfile = "annotationTxt.txt"
	with open(outputfile, "a+", encoding='utf-8') as f:
		comments = df.iloc[:, 10].values
		for comment in comments:
			if comment != "":
				comment = comment.strip()
				f.write(str(comment) + "\n")

if __name__ == '__main__':
	createAnnotationCSV()