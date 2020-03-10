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

def compute_result(y_pred,y_true):
    df = pd.DataFrame(columns=['y_pred','y_true'])
    for i,j in zip(y_pred,y_true):
        df = df.append({'y_pred': i , 'y_true': j},ignore_index=True)


    df.to_csv('result.csv',sep = "\t",index_label=None , header= True,index=False)
    return df


def compute_accuracy(result):
    true_predictions = 0
    false_predictions = 0
    for index, row in result.iterrows():
        if row['y_true'] == 1:
            if row['y_pred'] > 0.5:
                true_predictions += 1
            else:
                false_predictions += 1
        else:
            if row['y_pred'] < 0.5:
                true_predictions += 1
            else:
                false_predictions += 1

    return true_predictions,false_predictions,(true_predictions/200)*100

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



def input_preprocessing(input):

    q = re.sub('[?.,#!:;"''`]', '', input)
    q = q.strip()
    q = q.lower()
    tokenized = nltk.word_tokenize(q,language='english')

    ret_value = []
    for word in tokenized:

        if word in vocabulary:
            ret_value.append(vocabulary[word])
        else:
            print("Word not in vocabulary!")

    return ret_value


def zero_padding(top100_vectors):
    transformed_vec = []
    for vec in top100_vectors:
        if len(vec) < max_seq_length:
            var = []
            difference = max_seq_length - len(vec)
            for i in range(difference):
                var.append(0)

            temp = var + vec
            transformed_vec.append(temp)
        else:
            return transformed_vec

    return transformed_vec



index_questions = joblib.load('serialized/questions_dict')
index_questions_paraphrased = joblib.load('serialized/questions_paraphrased_dict')

vocabulary,inverse_vocabulary = make_vocabulary()


df = pd.DataFrame(columns=['question1','question2','similar'])
input_data = make_input_data(df)


questions_cols = ['question1','question2']

X_train = input_data[questions_cols]
Y_train = input_data['similar']


# Convert labels to their numpy representations
Y_train = Y_train.values


#split to dicts
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



input_question = joblib.load('serialized/input_question')
top100_q = joblib.load('serialized/top100_questions')
top100_a = joblib.load('serialized/top100_answers')


return_value = input_preprocessing(input_question)


if len(return_value) < max_seq_length:
    temp = max_seq_length - len(return_value)
    var = []
    for i in range(temp):
        var.append(0)
    ret_value = var + return_value



#ret_value = np.asarray(ret_value)
#transformed_vec = np.asarray(transformed_vec)


# Zero padding
for dataset, side in itertools.product([X_train], ['question1', 'question2']):
    dataset[side] = pad_sequences(dataset[side],  maxlen=max_seq_length)



embedding_dim = 50# This will be the embedding matrix
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
embeddings[0] = 0  # So that the padding will be ignored


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

top100_vectors = []
for q in top100_q:
    top100_vectors.append(input_preprocessing(q))


y_pred = []
for vec in top100_vectors:
    df = pd.DataFrame(columns=['question1', 'question2'])
    df = df.append({'question1': return_value, 'question2': vec}, ignore_index=True)

X_predict = {'question1':df.question1 , 'question2':df.question2}

for dataset, side in itertools.product([X_predict], ['question1', 'question2']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)


y_pred = malstm.predict([X_predict['question1'], X_predict['question2']])

print(y_pred)
