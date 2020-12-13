import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix


df = pd.read_csv('dataStep1.csv',delimiter=',',encoding='latin-1',  header = None)
# remove the context col

print(df.head())

# plot the SARCASM and NOT_SARCASM
# sns.countplot(df[0])
# plt.xlabel('Label')
# plt.title('Number of S and NONS')
# plt.show()
X = df[1] + " " + df[2]
Y = df[0]
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,)
print('below is X')
print(X.shape)
print(Y.shape)
print("Finish input data")

# for test
df0 = pd.read_csv('dataStep1Test.csv',delimiter=',',encoding='latin-1',  header = None)
X_test = df0[1] + " " + df0[2]
frames = [X, X_test]
X = pd.concat(frames)
# print(X)

# use tf-idf-document length get a n * 3000 dimension words
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 3, max_df = 0.5)
corpus = list(X.values)
X = tfidf.fit_transform(corpus)
X = X.todense()



# split the data

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

X_train, X_test = train_test_split(X, shuffle=False, test_size = 0.26470)


print(X_train.shape, X_test.shape)
# X_train = X
Y_train = Y



clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, Y_train)

print("Finish the fit model")

y_pred = clf.predict(X_test)

# cm = confusion_matrix(Y_test, y_pred)
#
# print("Finish training")
#
# print(cm)
# from sklearn.metrics import classification_report
# print(classification_report(Y_test, y_pred))


print("Finish training")
test_label = y_pred
answer = []
for i in range(len(test_label)):
    if (test_label[i] > 0.5):
        answer.append("twitter_" + str(i + 1) + ",SARCASM")
    else:
        answer.append("twitter_" + str(i + 1) + ",NOT_SARCASM")

with open('answer.txt', 'w') as f:
    for item in answer:
        f.write("%s\n" % item)
