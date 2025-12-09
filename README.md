# CSCI-335-03-Group-03

Abstract: Our project aims to identify the
sentiment of various tweets. We accomplish this goal
using SVM and BERT models. This allows us to
compare the effectiveness of these different models at NLP sentiment
analysis.

Developers: 

Cameron Kline:

- data collection
- BERT model
- CLI to train models
    
Edison Pan:

- data preprocessing
- SVM model
- CLI to test models


To run the models use the cli.py
Usage is python code/cli.py svm|bert train [train_path valid_path]
or: python code/cli.py svm|bert train [valid_path]

Example: python code/cli.py train svm
or python code/cli.py test bert

Make sure model_type is svm or bert, and that both input_path and output_path are .csv
the input csv should have 4 columns each being "tweet_id", "entity", "sentiment", "tweet_content"
the output csv will be the same as input csv but will add a new column called predicted,
which will give the predicted sentiment of the tweet.

Based off model_type the program will use the model in best_svm_pipeline.pkl or best_bert_pipeline.pkl
which are located in the data directory to predict the input_file given. Then it will give save the 
results to the output_file which will be located in the data directory. 
To make things easier a sample test input file is given as data/twitter_validation.csv. However, this is
the same validation data that we used for validation when doing grid search on our models. Thus
results will reflect performance on previously seen validation data instead of new unseen data if
using this file.  After the output is saved, evaluation metrics will also be printed out as well as
classification report
