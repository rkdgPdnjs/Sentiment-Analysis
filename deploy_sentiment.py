# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:22:29 2022

@author: Alfiqmal
"""

import re
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Embedding
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sentiment_analysis_modules import ExploratoryDataAnalysis
import os
import datetime
import json

#%% Model Loading

MODEL_PATH = os.path.join(os.getcwd(), "model.h5")
JSON_PATH = os.path.join(os.getcwd(), "tokenizer_data.json")

sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary()

#%% Tokenizer Loading



with open(JSON_PATH, "r") as json_file:
    token = json.load(json_file)



#%% Deploy

#new_review = ["<br \> I think the first one hour is interesting but \
#              the second half of the movie is boring. this movie just wasted\
#                 my precious time and hard earned money.<br />"]
  
new_review = [input("Review about this movie \n")]
#%%

# STEP 1: Data Cleaning

eda = ExploratoryDataAnalysis()
removed_tags = eda.remove_tags(new_review)
cleaned_input = eda.lower_split(new_review)

# STEP 3: Features Selection


# STEP 4: Data Preprocessing

loaded_tokenizer = tokenizer_from_json(token)

new_review = loaded_tokenizer.texts_to_sequences(cleaned_input)
new_review = eda.sentiment_pad_sequence(new_review)

#%% Model Prediction

outcome = sentiment_classifier.predict(np.expand_dims(new_review, axis = -1))

sentiment_dict ={1: "positive", 0: "negative"}
print("This review is " + sentiment_dict[np.argmax(outcome)])