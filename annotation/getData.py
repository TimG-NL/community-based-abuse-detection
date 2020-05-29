import praw
import pandas as pd
import numpy as np
import copy


def get_parents(reddit, data):
	# Get posts from subreddits
	output_dict = {"parent_id": [], 
					"child_id": [],
					"subreddit": [],
					"parent_text": [], 
					"child_text": []
					}


	# Get comments and add them to corresponding posts
	counter = 0
	for index, row in data.iterrows():
		if counter >= 4700:
			# Get child data
			child_id = row['comment_id']
			child_text = row['text']
			subreddit = row['subreddit']

			# Get parent data
			try:
				parent_submission = reddit.comment(id=child_id).parent()
				parent_id = parent_submission.id
				parent_text = parent_submission.body
			except:
				parent_id = np.nan
				parent_text = np.nan
			
			# Add to relevant columns
			output_dict["parent_id"].append(parent_id)
			output_dict["child_id"].append(child_id)
			output_dict["subreddit"].append(subreddit)
			output_dict["parent_text"].append(parent_text)
			output_dict["child_text"].append(child_text)
			#print("{}\t{}\t{}\t{}\t{}".format(parent_id, child_id, subreddit, parent_text, child_text))
			#print(comments_dict)


		if counter >= 5700:
			break

		counter += 1
		print(counter)

	
	output = pd.DataFrame(output_dict)
	output.to_csv("newStudentAnnotations5700.csv", sep='\t')
	print(output.head())
	print(output.shape)
	


def get_student_files(studentData):
	# Create your dictionary of the elements that you want to collect
	row_dict1 = {"parent_id": [], 
				"child_id": [],
				"subreddit": [],
				"parent_text": [], 
				"child_text": []
				}
	print(studentData.shape)
	print(studentData.columns)


	
	counter_group = 1
	counter_rows = 0
	student_counter = 1

	# Loop through all rows in single annotation file
	for index, row in studentData.iterrows():
		# Get duplicate data
		if counter_rows < 50:
			row_dict1["parent_id"].append(row.parent_id)
			row_dict1["child_id"].append(row.child_id)
			row_dict1["subreddit"].append(row.subreddit)
			row_dict1["parent_text"].append(row.parent_text)
			row_dict1["child_text"].append(row.child_text)
			
			counter_rows += 1
			
			if counter_rows == 50:
				row_dict2 = copy.deepcopy(row_dict1)
				row_dict3 = copy.deepcopy(row_dict1)
				row_dict4 = copy.deepcopy(row_dict1)
		# Get unique individual data
		elif counter_rows >= 50:
			if counter_rows < 150 and student_counter == 1:
				row_dict1["parent_id"].append(row.parent_id)
				row_dict1["child_id"].append(row.child_id)
				row_dict1["subreddit"].append(row.subreddit)
				row_dict1["parent_text"].append(row.parent_text)
				row_dict1["child_text"].append(row.child_text)
				counter_rows += 1
				if counter_rows == 150:
					student_counter += 1
			elif counter_rows >= 150 and counter_rows < 250 and student_counter == 2:
				row_dict2["parent_id"].append(row.parent_id)
				row_dict2["child_id"].append(row.child_id)
				row_dict2["subreddit"].append(row.subreddit)
				row_dict2["parent_text"].append(row.parent_text)
				row_dict2["child_text"].append(row.child_text)
				counter_rows += 1
				if counter_rows == 250:
					student_counter += 1
			elif counter_rows >= 250 and counter_rows < 350 and student_counter == 3:
				row_dict3["parent_id"].append(row.parent_id)
				row_dict3["child_id"].append(row.child_id)
				row_dict3["subreddit"].append(row.subreddit)
				row_dict3["parent_text"].append(row.parent_text)
				row_dict3["child_text"].append(row.child_text)
				counter_rows += 1
				if counter_rows == 350:
					student_counter += 1
			elif counter_rows >= 350 and counter_rows < 450 and student_counter == 4:
				row_dict4["parent_id"].append(row.parent_id)
				row_dict4["child_id"].append(row.child_id)
				row_dict4["subreddit"].append(row.subreddit)
				row_dict4["parent_text"].append(row.parent_text)
				row_dict4["child_text"].append(row.child_text)
				counter_rows += 1
			# Create and write dataframes to outputfiles
			if counter_rows == 450:
				output1 = pd.DataFrame(row_dict1)
				output2 = pd.DataFrame(row_dict2)
				output3 = pd.DataFrame(row_dict3)
				output4 = pd.DataFrame(row_dict4)
				print(output3.head(), output3.shape)
				print(output4.head(), output4.shape)
				print(output3.tail(), output3.shape)
				print(output4.tail(), output4.shape)
				output1.to_csv("group{}student{}.csv".format(counter_group, 1), sep="\t")
				output2.to_csv("group{}student{}.csv".format(counter_group, 2), sep="\t")
				output3.to_csv("group{}student{}.csv".format(counter_group, 3), sep="\t")
				output4.to_csv("group{}student{}.csv".format(counter_group, 4), sep="\t")

				# Reset counter_row and row_dict1 for next group
				row_dict1 = {"parent_id": [], 
				"child_id": [],
				"subreddit": [],
				"parent_text": [], 
				"child_text": []
				}
				counter_rows = 0
				counter_group += 1
				student_counter = 1
		


#Reddit Credentials
reddit = praw.Reddit(client_id='9kM6FEuQ7wySEA',
					 client_secret='IiV36X4RNTW_KGNLCLVpo5-ovDw',
					 user_agent='top100subredditchecker',
					 username='oneHelpfulPerson',
					 password='#')

#data = pd.read_csv('annotation.csv')
studentData = pd.read_csv('newStudentAnnotations.csv', sep="\t", header=0)

#get_parents(reddit, data)
get_student_files(studentData)
