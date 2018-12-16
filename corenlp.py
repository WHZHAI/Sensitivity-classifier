# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:59:02 2018

@author: zzzwh
"""
import backOff as bo
import spacy
import json
from stanfordcorenlp import StanfordCoreNLP


'''
this file will return the dictinoary of relation triples for each document,
no input needed but the dictionary of all text should be available for this file,
the 'text_dict' is the variable name for the text dictionary,
this file depend on the 'backOff' file
(if turn this file into a function, it would works better)
'''


nlp = spacy.load('en_core_web_sm')

'''
need the stanford core nlp program then pass the path into the wrapper
'''
snlp = StanfordCoreNLP(r'path of corenlp')


z = 0 # counter
sen_rel_dict = dict()

# property needed for corenlp
props={'annotators': 'openie',\
           'openie.triple.strict':'true',\
           'outputFormat':'json'}
    
'''
need the dictionary of all text, key is record name value is record text
'''
for k,v in text_dict.items():
    z+=1
    print(z)
    doc = nlp(v)
    doc_rel_list = []
    for s in doc.sents:
        output = snlp.annotate(s.text,properties=props)
        sen_rel_list = []
        try:
            data = json.loads(output)
            #extract openie
            res = [item["openie"] for item in data['sentences']]
                
            for i in res:
                for rel in i:
                    sub,ob = bo.bo_sub_obj(s.text,rel['subject'],rel['object'])
                    relation = rel['relation'].lower(),sub,ob
                    sen_rel_list+=[relation]
        except:
            sen_rel_list+=[]
        doc_rel_list += sen_rel_list
    sen_rel_dict[k]=doc_rel_list

snlp.close()
