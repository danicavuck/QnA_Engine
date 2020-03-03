import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import wordnet
import csv
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
                #sent = re.sub('[?.,#!:;"''`]', '', row[1])
                #sent = sent.lower()
                index_answers[line_count - 1] = row[2]
                index_question[line_count - 1] = row[1]
                line_count += 1


    return index_answers,index_question


index_answers , index_question = parse_csv_dataset()

lemmatizer = WordNetLemmatizer()
#text_sentence_tokens = nltk.sent_tokenize(text)


sentences = []
for sent in index_question.values():
    sentence = lemmatize_sentence(sent.lower())
    sentence = re.sub('[?]', '', sentence)
    sentence = sentence.strip()
    sentences.append(sentence)


Tfidf_Vec = TfidfVectorizer(tokenizer=Tokenize, stop_words='english')
tfidf = Tfidf_Vec.fit_transform(sentences)

print("Ask your question:")
input_question = input()
joblib.dump(input_question,'input_question')
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
joblib.dump(questions,"top100_questions")
joblib.dump(sentences,'question_corpus')
joblib.dump(answers,"top100_answers")