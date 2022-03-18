# CS 410 Final Project - Classification Competition

This is TEAM PYTHON's repo for CS 410 final project classification competition. This competition is about **Twitter Sarcasm Detection**. We implemented Naive Bayes, SVM, and LSTM to classify a Tweet as Sarcasm or Not sarcasm. LSTM model gave the best prediction for the test dataset.

Team members:
Chuanyue Shen (cs11),
Jianjia Zhang (jianjia2),
Runpeng Nie (runpeng3)

## Prerequisite

Please use Python3 and install the following packages:  

- numpy
- jsonlines
- nltk
- string
- re
- autocorrect
- Keras
- sklearn
- Pandas


It is recommended to run our program on machines with GPU. 

## Run the code

### Data cleaning and preprocessing

Type

`python preProcessData.py`

### Train the model and make predictions

To run the LSTM model, type

`python lstm.py`

To run the SVM model, type

`python svm.py`

To run the Naive Bayes model, type  

`python navieBayes.py`

### Test dataset prediction

After running the model, the test dataset prediction will be saved in the local directory, named

`answer.txt`

## Reference
https://www.kaggle.com/kredy10/simple-lstm-for-text-classification

## Presentation
https://youtu.be/IC9ncGVvbcQ


## More details about the project

### Source code

Please refer to the Source code part to the "Run the code" part mentioned above. The test set prediction of our best results can be found in answer.txt. The F1 score of one of our best results using LSTM beat the baseline and can be found in the Livedatalab leaderboard under the name of cs11 and/or jianjia2.

### Implementation details

#### Data preprocessing:
Training data and test data store in JSON line file. Each data item has three fields. The first field of the training data is a label, indicating whether the response of this data is sarcasm or not. The first field of the test data is its ID. Both training data and test data have a response field and a context field. The response field stores the tweet to be classified, and the context field stores the conversation context of the response relatively.
Both response and context are string data. Because the data is the tweets, so there are lots of emoji objects and @USER marks. The first step is removing the emojis and @USER. First of all, the regular expression package is used since all emojis are encoded in Unicode. By using re.compile() function the emoji Unicode pattern is defined, and emojis are removed through re.sun() function.
@USER is relatively easier to remove. Just like what we learned from lectures, there are lots of meaningless stop words like "the", "a" etc, they are almost useless for text classification. The NLTK package provides an English stopwords list, by adding "@USER" to the list and removing all words that appear in the list from response and context. Besides, for text classification, the punctuation character is useless too since the punctuation does not hold sentiment like normal words. Words are combined except punctuation through using string.punctuation. After these operations, the data is cleaned.
Only having a clean dataset is not enough. In English, with the change of grammar and context, there are many words with similar roots that have a similar meaning. For instance computer, computing, and computational. Their appearance increases the complexity of matrix operations. To improve performance and raise classification accuracy, words need to be grouped/replaced by their stem word. NLTK's PortStemmer tool is very useful. By replacing some words with their stem word, the computation complexity dropped significantly.

With the steps mentioned above, we can generate a n * 3 numpy array. The 3 columns store label, response, and context, respectively. The array is saved in 'dataStep1.csv' file.

Using the n * 3 numpy array, we can build a term document matrix that counts the frequency of each vocabulary in each tweet. The term document matrix is saved in 'term_doc_matrix.csv' file.

'dataStep1.csv' is mainly used in LSTM model implementation with some further processing.

'term_doc_matrix.csv' is mainly used in Naive Bayes and SVM models.

#### Model implementation and test result
• Naive Bayes

For this model, we choose the Gaussian distribution to fit the distribution of the count of each word. Then use the Naive Bayes classifier to fit the data. 

Result:
▪ Precision = 0.5371
▪ Recall = 0.7967
▪ F1 = 0.6416

• SVM

For this model, we preprocess the data using TF-IDF methods. Due to the reason that we have too many unique words and which will result in around 20,000 features, but the training data has only 5000 rows which will possibly result in underfitting. We remove the words with term count less than 3. After the data cleaning, the unique words are around 6,000, which is reasonable compared to original unique words. 

In the training process, we shuffle the training dataset and split it as training data (0.8) and validation data (0.2). 

Result:
▪ Precision = 0.5829
▪ Recall = 0.8633
▪ F1 = 0.6959

• LSTM (best model)

For this model, we create a neural network with LSTM, and word embeddings were learned while fitting the neutral network. We try to manipulate the number of layers and layer depths to find the best model architecture. Our final best RNN with LSTM model is composed of an embedding layer, a LSTM layer, 3 densely-connected layers, and activation and dropout layers in between. 'ReLu' is used for the activation function. Dropout = 0.5 is used to reduce overfitting. We adopt Sigmoid as the activation function in the output layer. 

In the training process, we try to compare different loss functions (mse, binary cross entropy), optimizers (Adam, SGD, RMSprop), and batch sizes (16,32,64,128). Our final best model using binary cross entropy as the loss function, RMSprop as the optimizer, and batch size = 128. Also, we shuffle the training dataset and split it as training data (0.8) and validation data (0.2).

For the data input, we further process the 'dataStep1.csv'. At first, we take the 5000 strings as the input and for each string, we choose at most max_len = 100 words from the right side to the left side. And then, based on the vocabulary, we store the index of each word for each string in the table (matrix_sequences) with its size as 5000 by 100. Overall, the data input for the LSTM model would be a matrix with a size of 5000 by 100.

Result
▪ Precision = 0.6068
▪ Recall = 0.8967
▪ F1 = 0.7238

### Contribution

All team members made equal contribution to the project and commited 20 hours+ per person to this project.
