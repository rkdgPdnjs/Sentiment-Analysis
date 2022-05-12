# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:45:33 2022

@author: Alfiqmal
"""

import re
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Embedding
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import datetime
import json

#%%

class ExploratoryDataAnalysis():
    
    def __init__(self):
       
        pass
    
    def remove_tags(self, data):
        '''
        To remove html tags from strings

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        '''
        
        for index, text in enumerate(data):
            # data[index] = text.replace("<br />", "") # not robust
            data[index] = re.sub("<.*?>", "", text) # robust (good)
        
        return data
    
    def lower_split(self, data):
        '''
        To lower-case all strings and split them

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        '''
        
        for index, text in enumerate(data):
            data[index] = re.sub("[^a-zA-Z]", " ", text).lower().split()
            
        return data
    
    def sentiment_tokenizer(self, data, token_save_path, num_words = 10000,
                               oov_token = "<OOV>", prt = False):
        '''
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        token_save_path : TYPE
            DESCRIPTION.
        num_words : TYPE, optional
            DESCRIPTION. The default is 10000.
        oov_token : TYPE, optional
            DESCRIPTION. The default is "<OOV>".
        prt : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        '''
        
        tokenizer = Tokenizer(num_words = num_words,
                              oov_token = oov_token)
        tokenizer.fit_on_texts(data)

        # To save tokenizer for deployment purpose

        
        token_json = tokenizer.to_json()



        with open(TOKENIZER_JSON_PATH, "w") as json_file:
            json.dump(token_json, json_file)

        # to observe number of words

        word_index = tokenizer.word_index
        #print(word_index)
        print(dict(list(word_index.items())[0:10]))

        # vectorize the sequence of text

        data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def sentiment_pad_sequence(self, data, maxlen = 200, 
                                   padding = "post",
                                   truncating = "post"):
        '''
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        maxlen : TYPE, optional
            DESCRIPTION. The default is 200.
        padding : TYPE, optional
            DESCRIPTION. The default is "post".
        truncating : TYPE, optional
            DESCRIPTION. The default is "post".

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        '''
        
        data = pad_sequences(data,
                             maxlen = 200,
                             padding = padding,
                             truncating = truncating)
        
        return data
    
    def sentiment_OneHotEncoder(self, data):
        '''
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        ohe = OneHotEncoder(sparse = False)
        data = ohe.fit_transform(np.expand_dims(data, axis = -1))
        
        return data

class ModelCreation():
    
    def LSTM_layer(self, num_words, nb_categories, embedding_output = 64,
                       nodes = 32, dropout = 0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences = True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation = "softmax"))

        model.summary()
        
        return model

    def simple_LSTM_layer(self, num_words, nb_categories, embedding_output = 64,
                              nodes = 32, dropout = 0.2):
        
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        model.add(Bidirectional(LSTM(nodes, return_sequences = True)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation = "softmax"))

        model.summary()
        
        return model
    
class ModelEvaluation():
    
    def report_metrics(self,y_true,y_pred):
       
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))

#%%



#%%
if __name__ == 'main':
    LOG_PATH = os.path.join(os.getcwd(),'log')
    MODEL_SAVE_PATH = os.path.join(os.getcwd(), "model.h5")
    TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), "tokenizer_data.json")
    URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
    
    df = pd.read_csv(URL)
    review = df["review"]
    sentiment = df["sentiment"]
    
    

    
    eda = ExploratoryDataAnalysis()
    test = eda.remove_tags(review)
    test = eda.lower_split(review)
    
    test = eda.sentiment_tokenizer(test, token_save_path = TOKENIZER_JSON_PATH,
                                   prt = True)
    
    test = eda.sentiment_pad_sequence(test)
    

    
    nb_categories = len(sentiment.unique())
    mc = ModelCreation()
    model = mc.LSTM_layer(10000, nb_categories)

