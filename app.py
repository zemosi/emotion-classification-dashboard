from flask import Flask, request, render_template, url_for
import re
import numpy as np
import pandas as pd
import community
import pandas as pd
import networkx as nx
import ast
import os
import csv
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import seaborn as sns
import ssl
try:
     _create_unverified_https_context =     ssl._create_unverified_context
except AttributeError:
     pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import nltk
nltk.download('stopwords')
sns.set(font_scale=1.2)
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle

app = Flask(__name__)
tt = TweetTokenizer()
PATH = os.getcwd()

alay_dict = pd.read_csv(PATH + '/riset-digital/dataset/new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original',
                                      1: 'replacement'})
stopwords_id = stopwords.words('indonesian')
stopwords_en = stopwords.words('english')

alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))


def tokenize_tweet(text):
    return " ".join(tt.tokenize(text))


def remove_unnecessary_char(text):
    text = re.sub("\[USERNAME\]", " ", text)
    text = re.sub("\[URL\]", " ", text)
    text = re.sub("\[SENSITIVE-NO\]", " ", text)
    text = re.sub('  +', ' ', text)
    return text


def preprocess_tweet(text):
    text = re.sub('\n', ' ', text)  # Remove every '\n'
    # text = re.sub('rt',' ',text) # Remove every retweet symbol
    text = re.sub('^(\@\w+ ?)+', ' ', text)
    text = re.sub(r'\@\w+', ' ', text)  # Remove every username
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)  # Remove every URL
    text = re.sub('/', ' ', text)
    # text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('  +', ' ', text)  # Remove extra spaces
    return text


def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])


def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text


def remove_stopword(text):
    text = ' '.join(['' if word in stopwords_id else word for word in text.split(' ')])
    text = ' '.join(['' if word in stopwords_en else word for word in text.split(' ')])
    text = re.sub('  +', ' ', text)
    text = text.strip()
    return text


def preprocess(text, alay=False, tweet=False):
    if (tweet):
        text = preprocess_tweet(text)
    text = remove_unnecessary_char(text)
    text = text.lower()
    text = tokenize_tweet(text)
    if (alay):
        text = normalize_alay(text)
    return text


# Cleaning the tweets
def clean(text):
    df_clean = pd.DataFrame()
    df_clean['tweet'] = text
    clean_text = df_clean['tweet'].apply(str).apply(preprocess, args=(True, True,))
    df_clean['clean'] = clean_text.replace("[^a-zA-Z#]", " ")
    df_clean['no_stopword_text'] = clean_text.apply(remove_nonaplhanumeric).apply(remove_stopword)

    return (df_clean[['clean', 'no_stopword_text']])


def scrap(count, query, since, until):
    # Setting variables to be used in format string command below
    tweet_count = count
    text_query = query + " lang:id"  # kata kunci yang digunakan untuk search di twitter
    PROJECT_NAME = "data"
    since_date = since
    until_date = until

    # Using OS library to call CLI commands in Python
    os.system(
        "snscrape --jsonl --max-results {} twitter-search '{} since:{} until:{}'> {}_scrap.json".format(tweet_count,
                                                                                                        text_query,
                                                                                                        since_date,
                                                                                                        until_date,
                                                                                                        PROJECT_NAME))

    data_df = pd.read_json(f'{PROJECT_NAME}_scrap.json', lines=True)

    # get username twitter from 'user'
    users = []
    # dataframe['column_name'] = dataframe['column_name'].fillna('').apply(str)
    user_prop = data_df['user'].fillna('').apply(str)
    i = 0
    for x in user_prop:
        i += 1
        if x != '' and re.search('({.+})', x) != None:
            dicti = ast.literal_eval(re.search('({.+})', x).group(0))
            users.append(dicti['username'])
        else:
            users.append('')

    data_df['username'] = users
    # Clearning
    data_df[['clean', 'no_stopword_text']] = clean(data_df['content'])

    # Load Data
    train = pd.read_csv(PATH + '/indonlu/dataset/emot_emotion-twitter/train_preprocess.csv')
    test = pd.read_csv(PATH + '/indonlu/dataset/emot_emotion-twitter/valid_preprocess.csv')

    train[['clean', 'no_stopword_text']] = clean(train['tweet'])
    test[['clean', 'no_stopword_text']] = clean(test['tweet'])

    data = pd.concat([train, test])

    tfidf_vect = TfidfVectorizer(max_features=5000)
    tfidf_vect.fit(data['no_stopword_text'])
    train_X_tfidf = tfidf_vect.transform(train['tweet'])
    test_X_tfidf = tfidf_vect.transform(test['label'])

    model = SVC(kernel='linear')
    model.fit(train_X_tfidf, train['label'])

    predictions_SVM = model.predict(test_X_tfidf)
    test['Prediction'] = predictions_SVM

    # Get accuracy score
    SVM_accuracy = accuracy_score(predictions_SVM, test['label']) * 100
    SVM_accuracy = round(SVM_accuracy, 1)

    model_path = PATH + '/TF-IDF'
    pickle.dump(model, open(model_path + 'emot_tf-idf_model.sav', 'wb'))
    pickle.dump(tfidf_vect, open(model_path + "emot_tfidf.pickle", "wb"))

    loaded_tfidf = pickle.load(open(model_path + 'emot_tfidf.pickle', 'rb'))
    loaded_model = pickle.load(open(model_path + 'emot_tf-idf_model.sav', 'rb'))

    # load dataset
    test_data = data_df

    # some preprocessing and setup
    test_data['no_stopword_text'].fillna('0', inplace=True)
    X_tfidf = loaded_tfidf.transform(test_data['no_stopword_text'])  # TF-IDF

    # Proses Pengujian
    predictions_SVM = loaded_model.predict(X_tfidf)
    test_data['prediction'] = predictions_SVM

    emotions = data_df["prediction"].value_counts().to_json(orient="records")
    category1 = data_df["prediction"].value_counts().to_json(orient="records")

    return emotions


@app.route('/', methods=['GET', 'POST'])
def index():  # put application's code here
    status = 0
    output = ''
    # If request method is POST, here
    if request.method == 'POST':
        status = 1
        form_data = request.form
        query = form_data.get('query')
        count = form_data.get('count')
        since = form_data.get('since')
        until = form_data.get('until')
        emotion_data = scrap(count,query,since,until)
        return render_template('index.html', output=output, status=status, emotion_data = emotion_data)
    # If request method is GET, here
    else:
        return render_template('index.html', status=status)


if __name__ == '__main__':
    app.run()
