# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:26:53 2018

@author: zzzwh
"""

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from itertools import compress
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import entropy

'''
This file shows the tfidf score for BORT features extracted, BORT pdframe produced by prepareTest.py is needed.
we could substitude BORT to other features then, all feature score list could be produced.
'''

tfidfvectorizer = TfidfVectorizer(analyzer='word',max_df=0.7)
index = tfidfvectorizer.fit(BORT_frame.value)
idf_value = index.idf_
vocab = index.vocabulary_ #vocab from all docs
cvectorizer = TfidfVectorizer(analyzer='word',vocabulary=vocab,use_idf=False) #normalised 

'''
this is for discrimitive score for sensitive triples, sensi_BORT_frame is frame for BORT extracted only from sensitive text
'''

sensi_tf = cvectorizer.fit_transform(sensi_BORT_frame.value)
sensi_array = sensi_tf.toarray()
sensi_tf_sum = sensi_array.sum(axis=0)
tfidf_value = np.multiply(sensi_tf_sum,idf_value)

#get needed term names
non0 = sensi_tf_sum!= 0 
feature_names = index.get_feature_names()

sensi_name = list(compress(feature_names,non0))
sensi_scores =list(compress(tfidf_value,non0))
final_scores = [(x,_) for x,_ in sorted(zip(sensi_name,sensi_scores),key= lambda pair: pair[1],reverse=True)]

'''
this is scores for all counts
all_finanl_scores is the score list we want
'''
all_tf = cvectorizer.fit_transform(rel_frame.value)
all_array = all_tf.toarray()
allarray_sum = all_array.sum(axis=0)
all_tfidf_value = np.multiply(allarray_sum,idf_value)

all_scores = list(compress(all_tfidf_value,non0))
all_final_scores = [(x,_) for x,_ in sorted(zip(sensi_name,all_scores),key= lambda pair: pair[1],reverse=True)]


'''
calculate KL Divergency

'''
kl = entropy(sensi_scores,all_scores) #more than 1, two distribution behave differently






