# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 23:46:15 2018

@author: zzzwh
"""

import nltk
from scipy import sparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update('u')

'''
this used to show evaluation summary and metric scores
'''
def eval_summary(predictions, labels):
  #test_predictions = model.predict(test_term_document_matrix)
  precision = precision_score(predictions, labels)
  recall = recall_score(predictions, labels)
  accuracy = metrics.balanced_accuracy_score(predictions, labels)
  f1 = fbeta_score(predictions, labels, 1) #1 means f_1 measure
  return precision,recall,accuracy,f1

'''
this is a tokenizer
'''
def tokenize(string):
    normalized_tokens = list()
    tokens = tokenizer.tokenize(string)
    for token in tokens:
        if token.lower() not in stop_words:
            normalized = token.lower()
            normalized_tokens.append(normalized)
    return normalized_tokens

'''
vectorizer for each feature, the ngrame_range could be changed so that creating unigram, bigram etc... 
'''
text_vectorizer = CountVectorizer(analyzer='word',tokenizer=tokenize,ngram_range = (1,2))
pos_vectorizer = CountVectorizer(analyzer='word',lowercase = False,ngram_range = (1,2))
BORT_vectorizer = CountVectorizer(analyzer='word')
DR_vectorizer = CountVectorizer(analyzer='word',lowercase = False)
DST_vectorizer = CountVectorizer(analyzer='word',lowercase = False)

'''
this step needs pd frame produced by the prepareTest file
'''
# create features matrix
text_matrix = text_vectorizer.fit_transform(text_frame['nlp_features']) 
pos_matrix = pos_vectorizer.fit_transform(pos_frame['nlp_features']) 
BORT_matrix = BORT_vectorizer.fit_transform(BORT_frame['nlp_features']) 
DR_matrix = DR_vectorizer.fit_transform(DR_frame['nlp_features'])
DST_matrix = DST_vectorizer.fit_transform(DST_frame['nlp_features'])

# stack features together
stacked_frame = sparse.hstack([text_matrix,pos_matrix,DR_matrix,BORT_matrix,DST_matrix])

# feature selection step
kbest = SelectKBest(score_func=chi2, k=22000)
new_matrix = kbest.fit_transform(stacked_frame,text_frame['class'])

# initialise classifier
lr = LogisticRegression(verbose=True,max_iter=800,class_weight='balanced',penalty='l2')
svc = SVC(kernel = 'linear',C=20)

# prepare for Kfold
kf = KFold(n_splits=5,shuffle=True) 
precision=0 
recall=0
acc=0
F1=0

'''
print k-fold evalutation result 

the code below marked down is used for mcnemar's test
'''
#result_array = np.zeros(3801)
for train_index, test_index in kf.split(new_matrix):
    
    X_train, X_test = new_matrix[train_index], new_matrix[test_index]
    y_train, y_test = text_frame['class'][train_index], text_frame['class'][test_index]
    
    svc_model = svc.fit(X_train, y_train)
    prediction_label = svc_model.predict(X_test)
    text_count = 0
#    for idx in test_index:
#        result_array[idx] = prediction_label[text_count]
#        text_count+=1
    p,r,a,f1 = eval_summary(svc_model.predict(X_test), y_test)
    precision+=p
    recall+=r
    acc+=a
    F1+=f1
print("Classifier '%s' has P=%0.5f R=%0.5f Acc=%0.5f F1=%0.5f" % \
      ('LR BoW text bigram',precision/5,recall/5,acc/5,F1/5))






















