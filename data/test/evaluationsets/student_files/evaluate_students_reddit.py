#!/usr/bin/python3

import pandas as pd
import numpy as np

from os import walk
from sklearn.metrics import cohen_kappa_score
from collections import Counter



def get_files():
	# Retrieve filenames per group
	f = []
	group_dict = {}
	for (dirpath, dirnames, filenames) in walk("./groups"):
		if len(filenames) > 0:
			group = dirpath.split("/")[2]
			group_dict[group] = sorted(filenames)
	return group_dict

def combine_groups(files_dict):
	groups_list = {}

	# Combine labels of groupmembers into one dataframe
	for group in sorted(files_dict.keys()):
		df_group = pd.DataFrame()

		for filename in files_dict[group]:
			df_annotations = pd.read_csv("./groups/{}/{}".format(group, filename), sep="\t")
			
			# Drop rows with Nan values for labels
			df_annotations.dropna(subset=['labels'], inplace=True)
			
			# Convert labels to numbers
			mapping_dict = {'Not': 0, 'Implicit' : 1, 'Explicit' : 2}
			df_annotations['labels'] = df_annotations['labels'].apply(lambda x: mapping_dict[x])

			# Only collect first 50 messages which the annotators have in common
			df_group[filename] = df_annotations['labels'].iloc[:50]

		groups_list[group] = df_group
	
	return groups_list
		
		
def calculate_kappa(group_list):
	counter = 1
	# for each group
	for group in group_list.keys():
		group_matrix = np.zeros((len(group.columns), len(group.columns)))
		# Compare each student with each other
		for i in range(len(group.columns)):
			print(group.columns[i])
			for j in range(i, len(group.columns)):
				#print(i, j, len(group.columns))
				student1 = group.iloc[0:50, i].values
				student2 = group.iloc[0:50, j].values
				# Calculate kappa score
				#print([print(i) for i in zip(student1, student2)])
				kappa_score = cohen_kappa_score(student1, student2, labels=[0,1,2])
				
				# Add scores to matrix
				group_matrix[i][j] = kappa_score.round(2)
				group_matrix[j][i] = kappa_score.round(2)


		#print(group_matrix)
		#if counter >= 2 and counter <= 4:
			
		#pd.DataFrame(group_matrix).to_csv("./output/group{}.csv".format(counter))
		counter += 1


def calculate_fleis_kappa(groups_list):
	"""
	Reference 
	https://en.wikipedia.org/wiki/Fleiss'_kappa#Calculations
	"""
	counter = 1
	fleiss_kappa_dict = {}

	# for each group
	for group in groups_list.keys():
		# Create numpy array with messages as rows and labels as columns
		label_counts_list = np.zeros((groups_list[group].shape[0], 3))
		row_counter = 0
		for i in groups_list[group].iterrows():
			label_counts = i[1].value_counts()
			# Add counts to rows in numpy array
			for label, count in label_counts.items():
				label_counts_list[row_counter][int(label)] = count
			row_counter += 1

		# # Calculate Fleiss Kappa for each group

		# Test example
		# n_array = [[0,0,0,0,14],
		# 			[0,2,6,4,2],
		# 			[0,0,3,5,6],
		# 			[0,3,9,2,0],
		# 			[2,2,8,1,1],
		# 			[7,7,0,0,0],
		# 			[3,2,6,3,0],
		# 			[2,5,3,2,2],
		# 			[6,5,2,1,0],
		# 			[0,2,2,3,7],
		# 			]



		# Calculate Fleiss Kappa for each group
		df = pd.DataFrame(label_counts_list)


		n_docs, n_labels = df.shape[0], df.shape[1]				# number of documents & labels
		n_annotators = df.iloc[0].sum()			# number of annotators

		# Calculate P_i
		p_iList = []
		for row in df.values:
			total = 0
			for label in row:
				#print(int(label))
				total += (label ** 2)
			#print(total)
			total -= n_annotators
			total /= (n_annotators * (n_annotators -1))
			p_iList.append(total)

		# Calculate P_j
		p_jList = []
		for label in df:
			p_jList.append(df[label].sum() / (n_docs * n_annotators))


		final_P_i = (1/n_docs) * sum(p_iList)
		final_P_j = sum([i**2 for i in p_jList])

		fleiss_kappa_score = (final_P_i - final_P_j) / (1 - final_P_j)
		fleiss_kappa_dict[group] = fleiss_kappa_score

	return fleiss_kappa_dict



def output_testset(files_dict, fleiss_dict):
	combined_list = []

	# Combine labels of groupmembers into one dataframe
	for group in sorted(files_dict.keys()):
		df_group = pd.DataFrame(columns=["group", "filename", "text", "a_1", "a_2", "a_3", "a_4", "a_5", "kappa"])

		counter_annotator = 1
		for filename in files_dict[group]:
			df_annotations = pd.read_csv("./groups/{}/{}".format(group, filename), sep="\t", index_col=0)
			
			# Drop Nan values
			df_annotations.dropna(subset=['labels'], inplace=True)


			# Convert labels to numbers
			mapping_dict = {'Not': 0, 'Implicit' : 1, 'Explicit' : 2}
			df_annotations['labels'] = df_annotations['labels'].apply(lambda x: mapping_dict[x])

			# Add common rows first with label of first annotator
			if counter_annotator == 1:
				common_messages = df_annotations.iloc[:50].values
				for row in common_messages:
					new_row = {'group': group,
							   'filename': group,
							   'text': row[4],
							   'a_1': row[5],
							   'kappa': fleiss_dict[group]
					}
					df_group = df_group.append(new_row, ignore_index=True)
			
			# Add the labels of the other annotators to existing data
			if counter_annotator > 1:
				common_messages = df_annotations.iloc[:50].values
				row_counter = 0
				for row in common_messages:
					   df_group.iloc[row_counter, 2 + counter_annotator] = row[5]
					   row_counter += 1

			
			# Add individual messages	
			individual_messages = df_annotations.iloc[50:].values
			for row in individual_messages:
				new_row = {'group': group,
						   'filename': filename,
						   'text': row[4],
						   'a_1'.format(counter_annotator): row[5],
						   'kappa': fleiss_dict[group]
				}
				df_group = df_group.append(new_row, ignore_index=True)
			

			counter_annotator += 1

		# Decide on final label
		labels = df_group[["a_1", "a_2", "a_3", "a_4", "a_5"]].values
		
		# Majority vote
		final_labels = [most_common(row) for row in labels]
		df_group['labels'] = final_labels

		# Combine data from all groups together
		[combined_list.append(row) for row in df_group.values]
	print([print(len(i)) for i in combined_list[140:160]])

	# Write data to a Dataframe
	all_groups_df = pd.DataFrame(combined_list, columns=["group", "filename", "text", "a_1", "a_2", "a_3", "a_4", "a_5", "kappa", "labels"])

	# Write to output csv file
	all_groups_df.to_csv("output/studentAnnotationData.csv", sep="\t")
	all_groups_df[["text", "labels"]].to_csv("output/test_students.csv", sep="\t")



def most_common(lst):
	# Returns the most common value within a list
    data = Counter(lst)
    return max(lst, key=data.get)


def main():
	# Retrieve files
	files_dict = get_files()

	# Combine student data for each group
	group_dfs = combine_groups(files_dict)

	# Calculate the kappa score for each group
	#calculate_kappa(group_dfs)

	# Calculate fleis kappa score
	fleiss_dict = calculate_fleis_kappa(group_dfs)

	# Apply Fleiss kappa to outputfiles
	output_testset(files_dict, fleiss_dict)

if __name__ == '__main__':
	main()