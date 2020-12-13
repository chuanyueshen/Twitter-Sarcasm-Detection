import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Dropout, Input, Embedding, LSTM
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical

df = pd.read_csv('dataStep1.csv',delimiter=',',encoding='latin-1',  header = None)
# remove the context col

print(df.head())


X = df[1] + " " + df[2]
Y = df[0]
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)
print('below is X')
print(X.shape)
print(Y.shape)
print("finish input data")
# split the data

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

X_train = X
Y_train = Y
df0 = pd.read_csv('dataStep1Test.csv',delimiter=',',encoding='latin-1',  header = None)
X_test = df0[1] + " " + df0[2]


# set the limit for the total vocabulary
max_words = 5000
# set the limit for each response
max_length = 100
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X_train)
# sequences is a list of n list, in the try example, the lenth for each list is the len of each response, the value is the index of each word,
sequences = tokenizer.texts_to_sequences(X_train)
# print(sequences)
# sequences matrix is similar to the sequences, but each list with same length, which is build by start from the end of the each list in sequences
matrix_sequences = sequence.pad_sequences(sequences, maxlen = max_length)
# print(matrix_sequences)

# define RNN
def RNN():
    inputs = Input(name='inputs',shape=[max_length])
    layer = Embedding(max_words,50,input_length=max_length)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(128, name='FC0')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
print("finish set up model")
model.fit(matrix_sequences,Y_train,batch_size=128,epochs=50,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0)])
print("finish fit the model")
# process the test data
test_sequences = tokenizer.texts_to_sequences(X_test)
test_matrix_sequences = sequence.pad_sequences(test_sequences,maxlen=max_length)
print("finish testing")

# accr = model.evaluate(test_matrix_sequences,Y_test)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
# from sklearn.metrics import confusion_matrix
# y_pred = model.predict(test_matrix_sequences)
# y_pred =(y_pred>0.5)
# list(y_pred)
# cm = confusion_matrix(Y_test, y_pred)
# print(cm)
# from sklearn.metrics import classification_report
# print(classification_report(Y_test, y_pred))

test_label = model.predict_on_batch(test_matrix_sequences)
print("Finish training")
print(test_label)
answer = []
for i in range(len(test_label)):
    if (test_label[i] > 0.5):
        answer.append("twitter_" + str(i + 1) + ",SARCASM")
    else:
        answer.append("twitter_" + str(i + 1) + ",NOT_SARCASM")

with open('answer.txt', 'w') as f:
    for item in answer:
        f.write("%s\n" % item)
