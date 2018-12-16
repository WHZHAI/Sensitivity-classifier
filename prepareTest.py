# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 23:31:43 2018

@author: zzzwh
"""

import pandas as pd
import preProcessing as pre


'''
These functions provides pdframe needed for classification task
this file depend on the preProcessing file
'''
# this function need pass the feature dictionary and return corresbonding feature pdframe
def creat_frame(featuen_dict,file_class = pre.extract_class()):
    '''
    creat panda frame
    return: pandas dataframe
    '''
    feature_list = list()
    print('start creat text frame')
    for k,v in featuen_dict.items():
        k_rep = k.replace('.html.gz','')
        to_be_append = v
        feature_list.append((k,to_be_append,file_class[k_rep]))
    print('done')
    collabels = ['fileName', 'nlp_features', 'class']

    frame = pd.DataFrame(feature_list, columns=collabels)
    return frame

































