import numpy as np
import os
import re
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate, Embedding, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

def inputData(filename):
    inputs = []
    outputs = []
    with open(filename, "r") as fs:
        for line in fs:
            tmp = []
            tmp_str = line.split()
            inputs.append(str(tmp_str[10:]))
            tmp = re.findall(r'\d+', ' '.join(tmp_str[2:10]))
            total = int(re.findall(r'\d+', tmp_str[1])[0])
            tmp_outputs = []
            for i in range(len(tmp)):
                tmp_outputs.append(int(tmp[i]) / total)
            outputs.append(tmp_outputs)
    return inputs, outputs

def textPreparation(train_texts, train_scores, test_texts, test_scores):
    tokenizer = Tokenizer(num_words=2500)  # 建立一个2500个单词的字典
    tokenizer.fit_on_texts(train_texts)
    x_train = sequence.pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=250)
    x_test = sequence.pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=250)
    y_train = np.array(train_scores)
    y_test = np.array(test_scores)
    return x_train, y_train, x_test, y_test

def text_cnn(maxlen = 250, max_features = 2500, embed_size = 64):
    texts_seq = Input(shape=[maxlen], name='x_seq')
    emb_texts = Embedding(max_features, embed_size)(texts_seq)
    convs = []
    filter_sizes = [2,3,4,5]
    for fsz in filter_sizes:
        conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_texts)
        pool = MaxPooling1D(maxlen - fsz + 1)(conv)
        flatten_pool = Flatten()(pool)
        convs.append(flatten_pool)
    merge = concatenate(convs, axis = 1)
    out = Dropout(0.5)(merge)
    hidden_output = Dense(64, activation='relu')(out)
    output = Dense(units=8, activation='softmax')(hidden_output)
    model = Model([texts_seq], output)
    model.compile(loss="MSE", optimizer="adam", metrics=['accuracy'])
    return model

def CNN_process(x_train, y_train):
    model = text_cnn()
    model.fit(x_train, y_train, validation_split=0.1, batch_size=128, epochs=20, shuffle=True)
    return model

def text_RNN(maxlen = 250, max_features = 2500, embed_size = 64):
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length = maxlen))
    model.add(Dropout(0.5))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(8, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

def RNN_process(x_train, y_train):
    model = text_RNN()
    early = EarlyStopping(monitor='val_acc', patience=5)
    model.fit(x_train, y_train, validation_split=0.1, batch_size=128, epochs=10, callbacks=[early], shuffle=True)
    return model

def printFunc(nntype, x_train, y_train, x_test, y_test, model):
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    print("\n"+nntype)
    print("train accuracy score", accuracy_score(y_train.argmax(axis=1), train_pred.argmax(axis=1)))
    print("test accuracy score", accuracy_score(y_test.argmax(axis=1), test_pred.argmax(axis=1)))
    print("fscore is: ", f1_score(y_test.argmax(axis=1), test_pred.argmax(axis=1), average='weighted'))
    print("Corr: ", pearsonr(y_test.argmax(axis=1), test_pred.argmax(axis=1)))

if __name__ == '__main__':
    train_texts, train_rates = inputData("./Lab_data/sina/sinanews.train")
    test_texts, test_rates = inputData("./Lab_data/sina/sinanews.test")
    x_train, y_train, x_test, y_test = textPreparation(train_texts, train_rates, test_texts, test_rates)
    
    cnn_model = CNN_process(x_train, y_train)    
    rnn_model = RNN_process(x_train, y_train)
    
    printFunc("CNN", x_train, y_train, x_test, y_test, cnn_model)
    printFunc("RNN", x_train, y_train, x_test, y_test, rnn_model)
