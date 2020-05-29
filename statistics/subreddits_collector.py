#!/usr/bin/python3

import pandas as pd
import csv
from nltk.tokenize import TweetTokenizer

years = ["2012", "2013", "2014", "2015", "2016", "2017"]
months= ["01", "04", "07", "10"]


outputfile = "used_subreddits.txt"
fieldnames = ["year", "month", 'comment_id', 'parent_text', 'subreddit', "timestamp", "gilded", "upvotes", "downvotes", "text"]
with open("{}".format(outputfile), "a+", encoding="utf-8") as f:
	subreddit_values = []
	for year in years:
		for month in months:
			print(year, month)
			inputfile = "reddish1sent_{}-{}.csv".format(year, month)
			df = pd.read_csv("../data/reddit/non-abusive/{}/{}".format(year, inputfile), header=None)

			comments = df.iloc[:, 4]
			comments.dropna(inplace=True)
			for subreddit in comments.values:
				subreddit_values.append(subreddit)
			
		



	sr = pd.Series(subreddit_values)
	sr.to_csv('subreddits.csv', header=['subreddit'])
	comment_values = list(set(subreddit_values))

	for comment in comment_values:
	  	f.write(str(comment) + "\n")