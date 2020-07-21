# Community-based Abuse Detection

In this repository you will find the code for the Master's Thesis: "Community-based Abuse Detection: Using Distantly Supervised Data and Biased Word Embeddings".

The goal of this thesis is to find out whether data coming from abusive communities and other non-abusive communities can be used for the detection of abusive language. 
The community-based data are collected from hateful communities and 'normal' subreddits on Reddit. With this data, we create distant datasets and generate task-specific polerized embeddings which are used to train abuse detection models. These models are tested both on an in-domain test set created in this research and on existing cross-domain test sets. 
This study confirms that data coming from abusive and non-abusive communities can be used for the detection of abusive language. The results indicate that models learn to classify abuse from silver distant training data (even though they still get outperformed by smaller gold training data). Furthermore, models that use pre-trained biased abusive embeddings generated from this data are showing competitive results when compared against much larger pre-trained generic embeddings.


# Data used in this research
## Training and Testing Data
- [AbusEval](https://github.com/tommasoc80/AbuseEval)
- [OffensEval 2019 Task A](https://competitions.codalab.org/competitions/20011)
- [OffensEval 2020 Task A](https://sites.google.com/site/offensevalsharedtask/results-and-paper-submission)


## Embeddings
- [GloVe Common Crawl (840B tokens)](https://nlp.stanford.edu/projects/glove/)
- [Abusive Reddit Embeddings](https://mega.nz/file/rVo33YZQ#reHxzIQduvYd33wF5xjp9OtWzvwa2--aDeWT4FsoAYo)
- [Non-abusive Reddit Embeddings](https://mega.nz/file/fVhhXICb#WMP-qmGxQ7icJy9pSlPnRk00DcAuCr4b8rmfyd20m80)


# Data Statement ([Bender and Friedman, 2018](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00041))

Data in the Gold Reddit test set has been collected from the medium Reddit and the language of the messages is English. The annotation of the explicit, implicit and non-abusive labels have been conducted by a group of first-year Information Science Bachelor students and the author of this paper. The annotators group consisted of 31 men and 10 women. Out of this group 21 students have previous experience with annotating documents and 19 students did not have any previous experience. The average age is 21.025 years and ranges between the ages of 18 and 42. All annotators have the Dutch nationality. 

All ages refer to the time of annotation: April 2020.


# Repository structure

### README.md
Description of the repository.
### requirements.txt
Required Python packages necessary for running the program.

### /annotations
Folder that contains code to create and extract annotated data by annotators.
### /annotations/output/
Folder that contains outputfiles of student annotations.
### /annotations/build_student_group_files.py
Creates annotation data for groups of students with common and individual messages.
### /annotations/collect_test_comments_students.py
Extracts test comments from Reddit archive files.
### /annotations/evaluate_students_annotations.py
Combines all annotations and calculates fleiss kappa within groups of annotators, writes final labels to output.

### /collection/extract_non_abusive
Folder that contains code to extract non-abusive messages from Reddit archive files.
### /collection/extract_non_abusive/collect_non-abusive_messages.py
Code to extract non-abusive messages from Reddit archive files.
### /collection/extract_non_abusive/expandedLexicon.txt
Lexicon used to filter explicit messages from abusive communities.
### /collection/extract_non_abusive/subreddit_statistics.tsv
List of abusive communities.

### /data
Folder that contains training and test data for the project.
### /data/training/
Folder that contains training data.
### /data/training/batches
Folder that contains the distant training data with 25-25-50 and 33-33-33 distributions of labels.
### /data/training/gold_train
Folder that contains the gold training data files: AbusEval and OffensEval2019.
### /data/test
Folder that contains all test data for the experiments.
### /data/clean_data_create_training_batches.py
Filters and creates distant training data sets and distant test sets.

### /models
Folder that contains LSTM and SVM folders.
### /models/lstm
Folder that contains code for building the Bi-LSTM models.
### /models/lstm/modelLSTM.py
Builds a Bi-LSTM model.
### /models/svm
Folder that contains code for building the SVM models.
### /models/svm/modelSVM.py
Contains code for SVM models and cross_validation experiments.

### /preprocessing
Folder that contains file for preprocessing and fastText input generation.
### /preprocessing/preprocessTrainingfiles_generateFasttextinput.py
File that cleans the reddit data and generates inputfiles necessary for the creation of fastText embeddings.

### /stats
Folder that contains extra data checks and analysis
### /stats/subreddits
Folder that contains text files that list which subreddits are used in which dataset.
### /stats/check_training_test_data.ipynb
File that checks the integrety and completeness of the datafiles
### /stats/reddit_abusive.ipynb
File that gathers statistics about the abusive data
### /stats/reddit_non-abusive_stats.ipynb
File that gathers statistics about the non-abusive data

### word_embeddings/fasttext 
Folder that contains the code for the creation and inspection of abusive and non-abusive word embeddings with fastText
### word_embeddings/fasttext/createEmbeddings.py
Code that creates the abusive and non-abusive embeddings with fastText
### word_embeddings/fasttext/inspectEmbeddings.py
Code that inspects the nearest neighbors of generated embeddings



# Usage of models via command line
## Experiment 1
### SVM
```$ python3 modelSVM.py <classification_type>-<exp_number>-<input_type>-<embeddings_source>-<distribution>-<batch_size>-<gold_data>```
  
```$ python3 modelSVM.py multiclass-1-tfidf-fasttext-252550-24000-NA```
  
### LSTM
```$ python3 modelLSTM.py <classification_type>-<exp_number>-<embeddings_source>-<distribution>-<batch_size>-<gold_data>```
  
```$ python3 modelLSTM.py binary-1-fasttext-333333-48000-NA```
  
  
## Experiment 2
### SVM
```$ python3 modelSVM.py <classification_type>-<exp_number>-<input_type>-<embeddings_source>-<distribution>-<batch_size>-<gold_data>```
  
```$ python3 modelSVM.py multiclass-2-embeddings-fasttext-NA-NA-offenseval2019```
  
### LSTM
```$ python3 modelLSTM.py <classification_type>-<exp_number>-<embeddings_source>-NA-NA-<gold_data>```

```$ python3 modelLSTM.py multiclass-2-glove-NA-NA-abuseval```


## Experiment 3
  ### SVM
```$ python3 modelSVM.py <classification_type>-<exp_number>-<input_type>-<embeddings_source>-<distribution>-<batch_size>-<gold_data>```
  
```$ python3 modelSVM.py binary-3-embeddings-glove-NA-NA-abuseval```
  ### LSTM
```$ python3 modelLSTM.py <classification_type>-<exp_number>-<embeddings_source>-<distribution>-<batch_size>-<gold_data>```
  
```$ python3 modelLSTM.py multiclass-3-fasttext-333333-12000-offenseval2019```

