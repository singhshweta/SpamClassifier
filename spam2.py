# -*- coding: utf-8 -*-
"""
Created on Wed May 30 21:17:30 2018

@author: shweta
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('spam.csv', encoding = "latin-1", header=None)
train_x, test_x, train_y, test_y = train_test_split(data[1], data[0])
vectorizer = TfidfVectorizer(stop_words='english',lowercase = True)
tfidf_train_x = vectorizer.fit_transform(train_x)

classifier = LogisticRegression()
classifier.fit(tfidf_train_x, train_y)
tfidf_test_x = vectorizer.transform(test_x)
#print tfidf_test_x.shape

scores = cross_val_score(classifier, tfidf_test_x, test_y, cv=5)
acc = scores.mean()
print( "Accuracy: %0.2f percent" % (acc *100))


mess = ['Congratulations, you have won $1000. email us your account details to claim the prize', "I am ready to invest in dating service"]
output = classifier.predict(vectorizer.transform(mess))

for i ,m in enumerate(mess):
	print( m, ' == ', output[i])
