import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
df_train = pd.read_csv('term_doc_matrix.csv', sep=',', header = None)
train_data = df_train.values
n = train_data.shape[0]
d = train_data.shape[1]
df_label = pd.read_csv('data_label.csv', sep=',', header = None)
train_label = df_label.values
train_label = train_label.reshape(n, )


df_test = pd.read_csv('term_doc_matrix_test.csv', sep=',', header = None)
test_data = df_test.values

print("Finish read data")
gnb = GaussianNB().fit(train_data, train_label)
test_label = gnb.predict(test_data)

print("Finish training")

answer = []
for i in range(len(test_label)):
    if (test_label[i] == 1):
        answer.append("twitter_" + str(i + 1) + ",SARCASM")
    else:
        answer.append("twitter_" + str(i + 1) + ",NOT_SARCASM")

with open('answer.txt', 'w') as f:
    for item in answer:
        f.write("%s\n" % item)
