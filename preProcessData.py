# import json
import numpy as np
import jsonlines
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# for preProcessingString
import re
from autocorrect import spell

# genrate stop-word dictionary inorder to remove useless words and charaters
stop_words = stopwords.words('english')
stop_words.append('user')
# stop_words = set(stop_words)
# exclude = set(string.punctuation)
# stop_words = stop_words.union(stop_words, exclude)
# stop_words = list(stop_words)

RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
dataStep1 = []
ps = PorterStemmer()

# pre precess string
def preProcessingString(text):
    # remove emoji
    text = RE_EMOJI.sub(r'', text)
    # lower
    text = ' '.join([word.lower() for word in word_tokenize(text)])
    # remove numbers, punctuation, strange character
    text = re.sub("[^a-zA-Z ]+", "", text)
    # stem and remove words in stopwords
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text)
    # spell check: correct the words like 'wrld' to 'world' (note: take longer time)
    # text = [spell(word) for word in (nltk.word_tokenize(text))]
    # text = ' '.join(text)

    return text

# get a n * 3 matrix from the original twitters
with jsonlines.open('data/train.jsonl') as reader:
    for item in reader:
        label = item["label"]
        response = preProcessingString(item["response"])
        context = ' '.join(str for str in item["context"])
        context = preProcessingString(context)
        dataStep1.append([label, response, context])

# get a 1800 * 2 matrix from test twitters
dataStep1Test = []
with jsonlines.open('data/test.jsonl') as reader:
    for item in reader:
        label = item["id"]
        response = preProcessingString(item["response"])
        context = ' '.join(str for str in item["context"])
        context = preProcessingString(context)
        dataStep1Test.append([label, response, context])

# https://www.kaggle.com/kredy10/simple-lstm-for-text-classification
dataStep1 = np.asarray(dataStep1)
print(dataStep1)
print(dataStep1.shape)
dataStep1Test = np.asarray(dataStep1Test)
np.savetxt("dataStep1.csv", dataStep1, delimiter = ',', fmt = '%s')
np.savetxt("dataStep1Test.csv", dataStep1Test, delimiter = ',',fmt = '%s')

# build the vocabulary
vocabulary = set()
vocabularyAll = set()
for eachLineAsList in dataStep1:
    vocabulary = vocabulary.union(eachLineAsList[1].split())
    vocabularyAll = vocabularyAll.union(eachLineAsList[1].split()).union(eachLineAsList[2].split())
vocabulary = list(vocabulary)
vocabularyAll = list(vocabularyAll)

#vocabulary size
# need to fix the vocabulary size as like 1000 or 2000, sort the words by the frequency
d = len(vocabulary)
dAll = len(vocabularyAll)

# train data size and test data size
n = len(dataStep1)
m = len(dataStep1Test)

data_label = np.zeros(n).astype('int8')
for i in range(n):
    if dataStep1[i][0] == 'SARCASM':
        data_label[i] = 1
print(data_label)

def build_term_doc_matrix(stringData, voc, isThisDataContainingContext):
    numberOfRows = len(stringData)
    numberOfCols = len(voc)
    newDataMatrix = np.zeros((numberOfRows, numberOfCols)).astype('int8')
    for i in range(numberOfRows):
        for j in range(numberOfCols):
            if (isThisDataContainingContext):
                newDataMatrix[i][j] = stringData[i][1].count(voc[j]) + stringData[i][2].count(voc[j])
            else:
                newDataMatrix[i][j] = stringData[i][1].count(voc[j])
    return newDataMatrix

# build two 5000 * d matrix for train data
term_doc_matrix = build_term_doc_matrix(dataStep1, vocabulary, False)
term_doc_matrix_All = build_term_doc_matrix(dataStep1, vocabularyAll, True)


# build two 1800 * d matrix for test data
term_doc_matrix_test = build_term_doc_matrix(dataStep1Test, vocabulary, False)
term_doc_matrix_All_test = build_term_doc_matrix(dataStep1Test, vocabularyAll, True)

# may need TF-IDF

print(vocabulary)
print(vocabularyAll)
print(term_doc_matrix)
print(term_doc_matrix.shape)
print(term_doc_matrix_All.shape)

print(term_doc_matrix_test)
print(term_doc_matrix_test.shape)
print(term_doc_matrix_All_test.shape)

# export matrix as csv file
np.savetxt("term_doc_matrix.csv", term_doc_matrix, delimiter = ',')
np.savetxt("term_doc_matrix_All.csv", term_doc_matrix_All, delimiter = ',')
np.savetxt("data_label.csv", data_label, delimiter = ',')
np.savetxt("term_doc_matrix_test.csv", term_doc_matrix_test, delimiter = ',')
np.savetxt("term_doc_matrix_All_test.csv", term_doc_matrix_All_test, delimiter = ',')
