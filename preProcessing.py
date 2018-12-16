# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:17:59 2018

@author: zzzwh
"""

import json
import gzip as gz
import os
from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer

'''
input: folder path of sensitive annotation
output: dict of sensitive text of annotation , key is file name, value is text
'''
def all_sensitive_text(origin_folder):
    allQuotes = dict()
    regex = re.compile(r'[^a-zA-Z]')
    for root, dirs, files in os.walk(origin_folder):
        #get the file names in the directory tree

        for file in files:
            directory = os.path.join(root,file)
  
            filename = file
            with open(directory,encoding ='utf-8') as f:
            # extract all sensitive ranges of a file
                quotes = str()# all ranges was saved in this list as strings
                for dic in f:
                    data = json.loads(dic)            
                    quote = data['quote']
                    quote = quote.replace('\n','')
                    quote = regex.sub(' ', quote)
                    quote = re.sub(r'\s{2,}',' ',quote)
                    quotes+=quote
                allQuotes[filename] = quotes
    return allQuotes

'''
input is origin folder path of all documents
output is a dictionary of all document, key is name ,value is processed text (tag removed) 
'''
def get_all_text(origin_folder):
    all_text = dict()
    for root, dirs, files in os.walk(origin_folder):
    #get the file names in the directory tree
        for file in files:
            target = os.path.join(root,file)
            file_name = file
            #print(target)
            #target_gz = directory + 'html.gz'
            with gz.open(target) as f:
                soup = BeautifulSoup(f, 'html.parser')
                pre_text = soup.pre.get_text()
                pre_text = pre_text.replace('\n','')
                #pre_text = re.sub(r'\s','',pre_text)
                pre_text = re.sub(r'^.*¶1\.','', pre_text)
                pre_text = re.sub(r'¶\d.','', pre_text)
                pre_text = re.sub(r'-{2,}','',pre_text)
                pre_text = re.sub(r'\s{2,}',' ',pre_text)
                pre_text = re.sub(r'¶.','',pre_text)

                all_text[file_name] = pre_text
    return all_text

'''
return a dict of file class: sensitive or non-sensitive
'''
def extract_class():
    '''
    return a dict with file name and file class.
    '''
    filecalss = dict()
    with open('full','r') as f:
        for line in f:
            l = line.split()
            filename = l[0].split('/')[2]
            cls = int(l[1])
            filecalss[filename] = cls
    return filecalss

