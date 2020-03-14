#!/usr/bin/python3

import json
import sys
import csv
from tqdm import tqdm
import bz2
from nltk import sent_tokenize
import re
from pprint import pprint
import pandas as pd



arg = sys.argv[1].strip('\n').split('-')
year, dataset = arg[0], arg[0] + "-" + arg[1]
input_file = "<path_to_file>{}}/RC_{}.bz2".format(year, dataset) # Example 2012/RC_2012-01.bz2
output_file = "out/{}/reddish_{}.csv".format(year, dataset) # Be sure to have empty files ready
output_file1 = "out/{}/reddish1sent_{}.csv".format(year, dataset)
output_file2 = "out/{}/reddish2sent_{}.csv".format(year, dataset)

# Load abusive subreddits
abusive_subreddits = []
with open("subreddit_statistics.tsv") as f:
	reader = csv.DictReader(f, delimiter="\t")
	for row in reader:
		if row["filter"] != "exclude":
				abusive_subreddits.append(row["subreddit"])


df = dict()
df1 = dict()
df2 = dict()
n = 0
child = dict()

year, month = dataset.split("-")
fieldnames = ["year", "month", 'comment_id', 'parent_text', 'subreddit', "timestamp", "gilded", "upvotes", "downvotes", "text", "abusive"]
with open(output_file, "a+") as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	#writer.writeheader()
	with bz2.open(input_file) as f:
		for line in tqdm(f):
			if n < 250000:
				try:
					comment = json.loads(line.strip())
				except:
					continue

				if not 'subreddit' in comment:
					continue

				subreddit = comment['subreddit']
				sentences = sent_tokenize(comment["body"])

				
				if subreddit.lower() not in abusive_subreddits:
					child[comment["parent_id"].split("_")[1]] = comment["id"]
					# Check for fields
					if 'retrieved_on' in comment:
						retrieved_on = comment['retrieved_on']
					else:
						retrieved_on = ""
					if not "downs" in comment:
						downs = 0
					else:
						downs = comment["downs"]
					if not "ups" in comment:
						ups = 0
					else:
						ups = comment["ups"]
					row = {
						"comment_id":comment['id'],
						"subreddit":comment['subreddit'],
						"timestamp":retrieved_on,
						"gilded":comment['gilded'],
						"upvotes":ups,
						"downvotes":downs,
						"text":comment['body']
						}

					df[comment["id"]] = row
					if row["text"] == "[deleted]":
						continue
					if len(sentences) <= 1:
						df1[comment["id"]] = row
					if len(sentences) <= 2:
						df2[comment["id"]] = row
					print(n)
					n += 1
					row["year"] = year
					row["month"] = month
					writer.writerow(row)
			else:
				break


with bz2.open(input_file) as f:
	for line in tqdm(f):
		n += 1
		try:
			comment = json.loads(line.strip())
		except:
			continue
		if comment["id"] in child:
			df[child[comment["id"]]]["parent_text"] = comment['body']
			if child[comment["id"]] in df1:
				df1[child[comment["id"]]]["parent_text"] = comment['body']
			if child[comment["id"]] in df2:
				df2[child[comment["id"]]]["parent_text"] = comment['body']



with open(output_file1, "a+") as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	#writer.writeheader()

	for comment_id, row in tqdm(df1.items()):
		if row["text"] == "[deleted]":
			continue
		row["year"] = year
		row["month"] = month
		writer.writerow(row)
with open(output_file2, "a+") as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	#writer.writeheader()

	for comment_id, row in tqdm(df2.items()):
		if row["text"] == "[deleted]":
			continue
		row["year"] = year
		row["month"] = month
		writer.writerow(row)