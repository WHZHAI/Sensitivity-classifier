# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 11:11:28 2018

@author: zzzwh
"""
import spacy
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import extract
import matplotlib.pyplot as plt
from spacy.lemmatizer import Lemmatizer


stemmer = SnowballStemmer('english')

nlp = spacy.load('en_core_web_sm')


'''
this function provide the back off for relation triples
'''

def bo_sub_obj(sentence,subj,obj):
    s = nlp(sentence)
    ents = list(s.ents)
    sub = []
    ob = []
    for e in ents:
        if e.text in subj:
            if e.label_ in ['GPE', 'ORG', 'PERSON', 'NORP', 'EVENT']:
                sub.append(e.label_)
        if e.text in obj:
            if e.label_ in ['GPE', 'ORG', 'PERSON', 'NORP', 'EVENT']:
                ob.append(e.label_)
    if not sub:
        sub.append('NP')
    if not ob:
        ob.append('NP')
    sub = '_'.join(sorted(sub))
    ob = '_'.join(sorted(ob))
    return sub,ob

                            
def rel_freq_frame(rel_dict, file_class = extract.extract_class()):  
    rel_list = []          
    for k,v in rel_dict.items():
        k_rep = k.replace('.html.gz','')
        rel_list.append((k,' '.join(v),file_class[k_rep]))
    collabels = ['key','value','class']
    frame = pd.DataFrame(rel_list,columns = collabels)
    return frame

  
    
    
    
    
    
    
    
    
    
    
    
    