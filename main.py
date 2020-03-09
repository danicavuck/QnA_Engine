import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import wordnet
import csv
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import joblib
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


def Tokenize(text):
    return nltk.word_tokenize(text.lower())


def parse_csv_dataset():
    index_answers = dict()
    index_question = dict()

    with open('insurance_qna_dataset.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                index_answers[line_count - 1] = row[2]
                index_question[line_count - 1] = row[1]
                line_count += 1


    return index_answers,index_question


def parse_paraphrased_dataset():
    index_question_paraphrased = dict()

    with open('paraphrazed_final_100.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                indeks = int(row[0]) * 5
                index_question_paraphrased[indeks] = row[1]
                line_count += 1
            else:
                indeks = int(row[0]) * 5
                index_question_paraphrased[indeks] = row[1]
                line_count += 1

    return index_question_paraphrased


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


def train_word2vec(train_input):
    model = Word2Vec(min_count=1)
    model.build_vocab(train_input)
    model.train(train_input, total_examples=model.corpus_count, epochs=model.iter)
    joblib.dump(model, 'serialized/word2vec')
    return model


index_questions_paraphrased =  parse_paraphrased_dataset()
index_answers , index_question = parse_csv_dataset()
lemmatizer = WordNetLemmatizer()
#text_sentence_tokens = nltk.sent_tokenize(text)


sentences = []
for sent in index_question.values():
    sentence = lemmatize_sentence(sent.lower())
    sentence = re.sub('[?]', '', sentence)
    sentence = sentence.strip()
    sentences.append(sentence)


train_input = corpus_preprocessing(sentences)

model = train_word2vec(train_input)


Tfidf_Vec = TfidfVectorizer(tokenizer=Tokenize, stop_words='english')
tfidf = Tfidf_Vec.fit_transform(sentences)

print("Ask your question:")
input_question = input()
question = re.sub('[?]', '', input_question)
question_lemmatized = lemmatize_sentence(question.lower())
question = nltk.sent_tokenize(question_lemmatized)
question_tfidf = Tfidf_Vec.transform(question)



nn = NearestNeighbors(n_neighbors= 100, algorithm='ball_tree').fit(tfidf)
distances,indices = nn.kneighbors(question_tfidf)
indices = indices.flatten()


questions = []
answers = []
for i in indices:
    q = index_question[i]
    a = index_answers[i]
    questions.append(q)
    answers.append(a)


print(questions)
joblib.dump(index_questions_paraphrased,'serialized/questions_paraphrased_dict')
joblib.dump(index_question,'serialized/questions_dict')
joblib.dump(questions,"serialized/top100_questions")
joblib.dump(sentences,'serialized/corpus')
joblib.dump(answers,"serialized/top100_answers")
joblib.dump(input_question,'serialized/input_question')
