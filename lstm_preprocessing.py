import datetime
import itertools
import numpy as np
import joblib
import nltk
from keras.optimizers import Adadelta
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import random
import re
from keras import backend as K, Model
import pandas as pd
#from main import corpus_preprocessing


from keras.layers import Input, Flatten, Dense, Dropout, Lambda, LSTM, Embedding
from sklearn.model_selection import train_test_split


def corpus_preprocessing(corpus):
    corpus = [word for word in corpus if word not in stopwords.words('english')]
    filtered_qs = []

    for sent in corpus:
        sent = re.sub('[?.,#!:;"''`]', '', sent)
        sent = sent.lower()
        filtered_qs.append(sent)

    train_input = []
    for q in filtered_qs:
        q = q.strip()
        word = q.split(" ")
        train_input.append(word)

    return train_input


def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


def make_vocabulary():
    vocabulary = dict()
    inverse_vocabulary = dict()
    index = 0
    for item in index_questions.items():
        q = re.sub('[?.,#!:;"''`]', '', item[1])
        q = q.strip()
        tokenized_sentence = nltk.word_tokenize(q)
        for word in tokenized_sentence:
            word = word.lower()
            if word not in vocabulary.keys():
                vocabulary[word] = index
                inverse_vocabulary[index] = word
                index += 1

    for item in index_questions_paraphrased.items():
        q = re.sub('[?.,#!:;"''`]', '', item[1])
        q = q.strip()
        tokenized_sentence = nltk.word_tokenize(q)
        for word in tokenized_sentence:
            word = word.lower()
            if word not in vocabulary:
                vocabulary[word] = index
                inverse_vocabulary[index] = word
                index += 1

    return vocabulary,inverse_vocabulary

def make_input_data(df):
    num_pos = 0
    num_neg = 0
    for item in index_questions.items():

        if (num_pos & num_neg) == 100:
            return df

        key_list_pos = []
        key_list_neg = []
        value_list_pos = []
        value_list_neg = []
        q = re.sub('[?.,#!:;"''`]', '', item[1])
        q = q.strip()

        sent_tokenized = nltk.word_tokenize(q)

        temp = item[0]
        if temp in index_questions_paraphrased:
            if num_pos < 100:
                for word in sent_tokenized:
                    word = word.lower()
                    if word in vocabulary:
                        key_list_pos.append(vocabulary[word])
                    else:
                        print("Word not in the dictionary")

                q_paraphrased = index_questions_paraphrased[temp]
                q = re.sub('[?.,#!:;"''`]', '', q_paraphrased)
                q_paraphrased = q.strip()
                q_tokenized = nltk.word_tokenize(q_paraphrased)
                for word in q_tokenized:
                    word = word.lower()
                    value_list_pos.append(vocabulary[word])


                if len(value_list_pos) != 0:
                   # key_list_pos = np.asarray(key_list_pos)
                   # value_list_pos = np.asarray(value_list_pos)
                    df = df.append({'question1':key_list_pos, 'question2':value_list_pos , 'similar':1},ignore_index=True)
                    num_pos += 1
            else:
               continue
        else:
            if num_neg < 100:
                for word in sent_tokenized:
                    word = word.lower()
                    if word in vocabulary:
                        key_list_neg.append(vocabulary[word])
                    else:
                        print("Word not in the dictionary")

                random_n = random.randrange(0, 27992)
                random_q = index_questions[random_n]
                q = re.sub('[?.,#!:;"''`]', '', random_q)
                q = q.strip()
                random_q_tokenized = nltk.word_tokenize(q)
                for word in random_q_tokenized:
                    word = word.lower()
                    value_list_neg.append(vocabulary[word])


                if len(value_list_neg) != 0:
                    #key_list_neg = np.asarray(key_list_neg)
                    #value_list_neg = np.asarray(value_list_neg)
                    df = df.append({'question1': key_list_neg ,'question2': value_list_neg ,'similar': 0},ignore_index=True)
                    num_neg += 1
            else:
                continue

    return

#def corpus_preprocessing(corpus):
    corpus = [word for word in corpus if word not in stopwords.words('english')]
    filtered_qs = []

    for sent in corpus:
        sent = re.sub('[?.,#!:;"''`]', '', sent)
        sent = sent.lower()
        filtered_qs.append(sent)

    train_input = []
    for q in filtered_qs:
        q = q.strip()
        word = q.split(" ")
        train_input.append(word)


    return train_input

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

index_questions = joblib.load('serialized/questions_dict')
index_questions_paraphrased = joblib.load('serialized/questions_paraphrased_dict')

vocabulary,inverse_vocabulary = make_vocabulary()


df = pd.DataFrame(columns=['question1','question2','similar'])
input_data = make_input_data(df)


validation_size = 100
training_size = len(input_data) - validation_size

questions_cols = ['question1','question2']

X_train = input_data[questions_cols]
Y_train = input_data['similar']


# Split to dicts
X_train = {'question1': X_train.question1, 'question2': X_train.question2}

# Convert labels to their numpy representations
Y_train = Y_train.values


Y_train = input_data['similar'].values
X_train = {'question1':input_data.question1,'question2':input_data.question2}





# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 50
epochs = 50

max_seq_length = max(input_data.question1.map(lambda x: len(x)).max(),
                     input_data.question2.map(lambda x: len(x)).max(),
                     input_data.question1.map(lambda x: len(x)).max(),
                     input_data.question2.map(lambda x: len(x)).max())


print(max_seq_length)


# Zero padding
for dataset, side in itertools.product([X_train], ['question1', 'question2']):
    dataset[side] = pad_sequences(dataset[side],  maxlen=max_seq_length)


embedding_dim = 50# This will be the embedding matrix
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
embeddings[0] = 0  # So that the padding will be ignored

print(embeddings.shape)

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=True)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(max_seq_length)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# Start training
training_start_time = datetime.time()

malstm_trained = malstm.fit([X_train['question1'], X_train['question2']], Y_train, epochs=epochs,verbose=1)


y_pred = malstm.predict([X_train['question1'], X_train['question2']])

tr_y = Y_train


tr_acc = compute_accuracy(tr_y,y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

