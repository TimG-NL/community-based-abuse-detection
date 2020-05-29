from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import pandas as pd

def loadData(test_set):
	# Get training data
	datafile = "../data/batches/5000/batch_abusive_0_10000_0.csv"
	train_df = pd.read_csv(datafile, names=['id', 'text', 'classes'], header = 0, index_col=0, error_bad_lines=False)
	train_df.dropna(inplace=True)
	mapping_dict = {'NOT': 0, 'IMP' : 1, 'EXP' : 2}
	train_df['labels'] = train_df['classes'].apply(lambda x: mapping_dict[x])
	print(train_df.head())

	# Get test data
	if test_set == 'olid':
		olid_df = pd.read_csv("../data/test/test_olid.csv", sep = "\t", header=0, index_col=0)
		print(olid_df.head())
		return train_df, olid_df
	elif test_set == 'hateval2019':
		pass
	elif test_set == 'hateval2020':
		pass
	elif test_set == 'offenseval2020':
		pass


def buildModel(train_df, test_df):
	# Load in the best model
	if os.path.isdir('output_RoBERTa/'):
	    model_RoBERTa = ClassificationModel('roberta', 'output_RoBERTa/', use_cuda=False)
	else:
		# Define the RoBERTa model
		parameters = {'num_train_epochs': 3, 
					'reprocess_input_data': False, 
					'output_dir': 'output_RoBERTa/', 
					'overwrite_output_dir': True, 
					"train_batch_size": 8,
					"learning_rate": 4e-5}

	    model_RoBERTa = ClassificationModel('roberta', 'roberta-base', num_labels=3, args=parameters, use_cuda = False)
	# # Train the model
	    model_RoBERTa.train_model(train_df)
	# # Evaluate the model
	    result, model_RoBERTa_outputs, wrong_predictions_RoBERTa = model_RoBERTa.eval_model(test_df, cr=classification_report, cm=confusion_matrix, acc=accuracy_score)
	# print(model_RoBERTa_outputs)
	    print(result['cr'])  # Classification Report
	    print(result['cm'])  # Confusion Matrix
	    print(result['acc'])  # Accuracy Score


def main():
	train_df, test_df = loadData('olid')
	buildModel(train_df, test_df)



if __name__ == '__main__':
	main()