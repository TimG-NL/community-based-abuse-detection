#!/usr/bin/python3

import pandas as pd
import numpy as np 
import redditcleaner
import emoji
import re
from sklearn.utils import shuffle


def filterText(row):
	text = str(row)
	# Filter URLs
	text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "URL", text)
	# Filter numbers
	text = re.sub(r'\b\d+\b', "NUMBER", text)
	# Filter usernames
	text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', "@USER", text)
	# Convert emojis to text
	text = emoji.demojize(text)
	# Clean reddit markup language
	text = redditcleaner.clean(text)

	return text


def hateval2019():
	# Labels : 
	# 0 = Hatespeech absent
	# 1 = Hatespeech present
	df_dev = pd.read_csv('evaluationsets/hateval2019/public_development_en/dev_en.tsv', names=['id', 'old_text', 'HS', 'TR', 'AG'], sep='\t')
	df_train = pd.read_csv('evaluationsets/hateval2019/public_development_en/train_en.tsv', names=['id', 'old_text', 'HS', 'TR', 'AG'], sep='\t')
	df_test = pd.read_csv('evaluationsets/hateval2019/reference_test_en/en.tsv', names=['id', 'old_text', 'HS', 'TR', 'AG'], sep='\t')
	#df_combined = df_train.append(df_dev, ignore_index=True).reset_index()
	#df_combined['text'] = df_combined['old_text'].apply(filterText)
	df_train['text'] = df_train['old_text'].apply(filterText)
	df_dev['text'] = df_dev['old_text'].apply(filterText)
	df_test['text'] = df_test['old_text'].apply(filterText)

	#df_combined['labels'] = df_combined['HS']
	df_train['labels'] = df_train['HS']
	df_train.drop(['old_text', 'HS', 'TR', 'AG'], inplace=True, axis=1)

	df_dev['labels'] = df_dev['HS']
	df_dev.drop(['old_text', 'HS', 'TR', 'AG'], inplace=True, axis=1)

	df_test['labels'] = df_test['HS']
	df_test.drop(['old_text', 'HS', 'TR', 'AG'], inplace=True, axis=1)
	
	df_train.to_csv('training/gold_train/hateval2019/train_hateval2019.csv', sep='\t')
	df_dev.to_csv('training/gold_train/hateval2019/dev_hateval2019.csv', sep='\t')
	df_test.to_csv('test/test_hateval2020.csv', sep='\t')


def olid():
	# Labels : 
	# 0 = Offensive language absent
	# 1 = Offensive language present
	df_train = pd.read_csv('evaluationsets/olid/olid_data.csv', names=['id', 'category', 'old_text'], header=1)
	df_train['text'] = df_train['old_text'].apply(filterText)
	df_train['labels'] = df_train.category.replace(('OFF', 'NOT'), (1, 0))
	df_train.drop(['old_text', 'category'], inplace=True, axis=1)


	df_test = pd.read_csv('evaluationsets/olid/olid_data.csv', names=['id', 'category', 'old_text'], header=1)
	df_test['text'] = df_test['old_text'].apply(filterText)
	df_test['labels'] = df_test.category.replace(('OFF', 'NOT'), (1, 0))
	df_test.drop(['old_text', 'category'], inplace=True, axis=1)

	df_train.to_csv('training/gold_train/olid/train_olid.csv', sep='\t')
	df_test.to_csv('test/test_olid.csv', sep='\t')


def offenseval2020():
	df = pd.read_csv('evaluationsets/offenseval2020/englishA-goldlabels.csv', sep=',', names=['id', 'old_text', 'category'], header=0)
	df['text'] = df['old_text'].apply(filterText)
	df['labels'] = df.category.replace(('OFF', 'NOT'), (1, 0))
	df.drop(['old_text', 'category'], inplace=True, axis=1)
	df.to_csv('test/test_offenseval2020.csv', sep='\t')


def abuseval():
	df_train = pd.read_csv('evaluationsets/abuseval/abuseval_offenseval_train.tsv', sep='\t', names=['id', 'old_text', 'offensive', 'offense_type', 'target', 'implicit_explicit', 'abuse'], header=0)
	df_test = pd.read_csv('evaluationsets/abuseval/abuseval_offenseval_test.tsv', sep='\t', names=['id', 'old_text', 'offensive', 'offense_type', 'target', 'implicit_explicit', 'abuse'], header=0)
	# df_combined = df_train.append(df_test, ignore_index=True).reset_index()
	# df_combined['text'] = df_combined['old_text'].apply(filterText)
	# df_combined.drop(['index', 'offense_type', 'target', 'implicit_explicit'], inplace=True, axis=1)
	# df_combined['labels'] = df_combined.abuse.replace(('NOTABU', 'IMP', 'EXP'), (0, 1, 2))
	# df_combined[['text', 'labels']].to_csv('evaluationsets/abuseval/test_abuseval.csv', sep='\t')

	df_train['text'] = df_train['old_text'].apply(filterText)
	df_test['text'] = df_test['old_text'].apply(filterText)

	df_train.drop(['offense_type', 'target', 'implicit_explicit'], inplace=True, axis=1)
	df_test.drop(['offense_type', 'target', 'implicit_explicit'], inplace=True, axis=1)

	df_train['labels'] = df_train.abuse.replace(('NOTABU', 'IMP', 'EXP'), (0, 1, 2))
	df_test['labels'] = df_test.abuse.replace(('NOTABU', 'IMP', 'EXP'), (0, 1, 2))

	df_train[['text', 'labels']].to_csv('training/gold_train/abuseval/train_abuseval.csv', sep='\t')
	df_test[['text', 'labels']].to_csv('test/test_abuseval.csv', sep='\t')

def student_files():
	df_students = pd.read_csv('../data/test/evaluationsets/student_files/output/test_students.csv', sep='\t', names=['id','old_text', 'labels'], header=0)
	df_self = pd.read_csv('../data/test/evaluationsets/self/self_annotations.csv', sep=',', names=['id', 'subreddit', 'text', 'distant_labels', 'labels'], header=0)

	df_students['text'] = df_students['old_text'].apply(filterText)
	df_students['annotator'], df_self['annotator'] = "students", "self"

	df_students.drop(['old_text'], inplace=True, axis=1)
	df_self.drop(['subreddit', 'distant_labels'], inplace=True, axis=1)
	

	# Collect abusive and calculate distribution
	df_self_abusive = df_self.loc[df_self['labels'].isin([1.0, 2.0])]
	df_students_abusive = df_students.loc[df_students['labels'].isin([1.0, 2.0])]
	df_combined_abusive = df_self_abusive.append(df_students_abusive, ignore_index=True).reset_index()
	
	# Collect non_abusive from students
	df_combined_non_abusive = df_students[df_students['labels'] == 0]

	# Combine abusive and non-abusive testsets with 33% and 66% distrubition of the labels
	df_final = df_combined_non_abusive.iloc[:df_combined_abusive.shape[0]*2][['text', 'labels', 'annotator']].append(df_combined_abusive[['text', 'labels', 'annotator']], ignore_index=True)

	df_final = shuffle(df_final, random_state=123)
	
	print(df_self.shape, df_self_abusive.shape)
	print(df_students.shape, df_self.shape, df_final.shape)
	df_final.to_csv('../data/test/test_students_self_reddit.csv', sep='\t')

def main():
	#hateval2019()
	#olid()
	#offenseval2020()
	#abuseval()
	student_files()


if __name__ == '__main__':
	main()
