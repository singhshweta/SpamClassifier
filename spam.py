# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:34:04 2018

@author: shweta
"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
#from wordcloud import WordCloud
from math import log, sqrt
import pandas as pd
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline




mails = pd.read_csv('spam.csv',encoding = 'latin-1')
#print(mails.head())

mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
#print(mails.head())

mails.rename(columns = {'v1': 'labels','v2': 'message'}, inplace = True)
#print(mails.head())

mails['label'] = mails['labels'].map({'ham':0,'spam':1})
mails.drop(['labels'],axis = 1, inplace = True)
#print(mails.head())
ham, spam = mails['label'].value_counts() 
tot_mails = ham+spam
#print(tot_mails)
trainIndex , testIndex = [],[]
for i in range(mails.shape[0]):
    if np.random.uniform(0,1) <0.75:
        trainIndex +=[i]
    else:
        testIndex +=[i]
train = mails.loc[trainIndex]
test = mails.loc[testIndex]

train.reset_index(inplace = True)
train.drop('index' ,axis = 1,inplace = True)
#print(train.head())

test.reset_index(inplace = True)
test.drop('index' ,axis = 1,inplace = True)
#print(test.head())

train_ham , train_spam = train['label'].value_counts()
#print(train_ham , train_spam)

test_ham , test_spam = test['label'].value_counts()
#print(test_ham , test_spam)
        
clf = MultinomialNB()
vectorizer = CountVectorizer()

messages = vectorizer.fit_transform(train['message'].values)
labels  = train['label'].values

clftf = TfidfTransformer()
messagestf = clftf.fit_transform(messages)
clf.fit(messagestf,labels)
pred = clf.predict(vectorizer.transform(test['message']))
print(np.mean(pred == test['label']))
#print(test[''])
## accuracy using pipeline 

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()), ])


text_clf = text_clf.fit(train['message'], train['label'])

predicted = text_clf.predict(test['message'])
print(np.mean(predicted == test['label']))

#print(pred)

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()), ])



#accuracy using NLTK
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
                      ('tfidf', TfidfTransformer()),
                      ('mnb', MultinomialNB(fit_prior=False)), ])

text_mnb_stemmed = text_mnb_stemmed.fit(train['message'], train['label'])

predicted_mnb_stemmed = text_mnb_stemmed.predict(test['message'])

print(np.mean(predicted_mnb_stemmed == test['label']))

























