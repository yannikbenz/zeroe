This is a subset of the original data to give you a hint about the structure

The original data has been taken from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Original files:
* train.csv - the training set, contains comments with their binary labels
* test.csv - the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.
* sample_submission.csv - a sample submission file in the correct format
* test_labels.csv - labels for the test data; value of -1 indicates it was not used for scoring; (Note: file added after competition close!)

To work properly with the data we joined test.csv ad test_labels.csv with respect to their ids and removed
all samples labeled with -1,-1,-1,-1,-1,-1.

The both resulting files are train.txt and test.txt

TC Labels:
* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

