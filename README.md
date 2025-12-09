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

Example: python code/cli.py svm test
or python code/cli.py bert train

To train, make sure model_type is svm or bert, and that both input_path and output_path are .csv
the input csv should have 4 columns each being "tweet_id", "entity", "sentiment", "tweet_content"
the output csv will be the same as input csv but will add a new column called predicted,
which will give the predicted sentiment of the tweet.

To test, choose a model. Then the program will use the model in best_svm_pipeline.pkl or best_bert_pipeline.pkl
which are located in the data directory to predict the input_file given. Testing will display the evaluation metrics,
as well as the classification report. The program will default to data/twitter_validation.csv if not specified.
However, this is the same validation data that we used for validation when doing grid search on our models. Thus
results will reflect performance on previously seen validation data instead of new unseen data if using this file.

Testing will also generate a report of the dataset with predicted sentiment. It will give save the results to the
output_file, either svm_prediction_output.csv or bert_prediction_output.csv, which will be located in the data directory. 

