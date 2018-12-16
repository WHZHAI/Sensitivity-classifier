# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:47:05 2018

@author: zzzwh
"""

import preProcessing
import spacy
from nltk import ngrams
import collections

nlp = spacy.load('en_core_web_sm')

def load_all_POS(text_dict):
    '''
    this function loads all POS sequences from text dictionary
    
    input: text dictionary

    return: a dictionary, key is file name, value is POS sequences
    '''

    pos_dict = dict()
    i = 1
    print('processing POS.....')
    for k,v in text_dict.items():
        pos_for_all_token = str()
        doc = nlp(v)
        for token in doc:
       
            pos_for_all_token+=' '+token.tag_
        pos_dict[k] = pos_for_all_token
        i+=1
        if i == len(all_text)//2:
            print('already half.....')
    print('Done')
    return pos_dict


def extract_DR(full_text):
'''
return the DR (depedency relations) feature

input : the dictionary of text

output: the dictionary of all features , key is file name,
        value is all features for this file
'''
    i = 0
    dep_dict = dict()
    for k,v in full_text.items():
        i += 1 
        if i == len(full_text)//2:
            print('already parse half')
        elif i == 2:
            print('it works')
        doc = nlp(v)
        spans = list(doc.ents) 
        sents = list(doc.sents)
        dep_str = str()
        for sen in sents:
            sub = sen.root.subtree
            for t in sub:
                if t.dep_ != 'punct' and t.dep_ != '':
                 dep_str += ' '+t.lemma_+t.head.lemma_+ t.dep_   
        dep_dict[k] = dep_str
    return dep_dict



'''
this function returns the DST (dependency subtree) feature of each record

input: text dictionary

output: dictionary of features for each record
'''
def extract_subtree(dic):
    iii = 0
    subtree_dict = dict()
    for key, item in dic.items():
        # if print of then this works
        iii+=1
        if iii == 3:
            print('ok')
        subtree = []
        doc = nlp(item)
        for sents in doc.sents:
            root = sents.root
            for child in root.children:
                if child.dep_ == 'prep':
                    for grand in child.children:
                        if grand.dep_ == 'pobj':
                            subtree.append(root.lemma_+'_'+grand.lemma_)
                if child.pos_ == 'NOUN' or child.pos_ == 'PROPN':
                    subtree.append (root.lemma_+'_'+child.lemma_)
    
        subtree_dict[key] = subtree 
    return subtree_dict

























