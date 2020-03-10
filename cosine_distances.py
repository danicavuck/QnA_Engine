import numpy as np
import joblib

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from scipy import spatial
import re



def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)

        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be','the']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))

    return results


def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec



def input_preprocessing(input_question):
    input = re.sub('[?.,#!:;"''`]', '', input_question)
    input = lemmatize_sentence(input.lower())
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(input)

    word_tokens = [w for w in word_tokens if not w in stop_words]

    final_input = ""
    for w in word_tokens:
        if w not in stop_words:
            final_input += " " + w

    final_input = final_input.strip()
    return final_input


input = joblib.load('serialized/input_question')
#input = "How Do I Get A Life Insurance License In Virginia?"
model = joblib.load('serialized/word2vec')
top100 = joblib.load('serialized/top100_questions')
answers = joblib.load('serialized/top100_answers')



lemmatizer = WordNetLemmatizer()



final_input = input_preprocessing(input)
print(answers)
#print(top100)
filtered_qs =[]
unfiltered_qs = []

for q in top100:
    unfiltered_qs.append(q)
    filtered_qs.append(input_preprocessing(q))


index2word_set = set(model.wv.index2word)
similarities = dict()

questions_dictionary = dict()
answers_dictionary = dict()
num1 = 0
num2 = 0

for q in unfiltered_qs:
    questions_dictionary[num1] = q
    num1 += 1

for a in answers:
    answers_dictionary[num2] = a
    num2 += 1



for s in questions_dictionary.items():
    value = s[1].strip()
    s1_afv = avg_feature_vector(final_input,model=model,num_features=100,index2word_set=index2word_set)
    s2_afv = avg_feature_vector(value, model=model,num_features=100,index2word_set=index2word_set)
    sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    similarities[s[0]] = sim



max_value = max(similarities,key=similarities.get)
final_q = questions_dictionary[max_value]
final_a = answers_dictionary[max_value]


print(final_q)
print(final_a)

