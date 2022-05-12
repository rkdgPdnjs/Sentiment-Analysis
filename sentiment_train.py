# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:46:19 2022

Python file that trains sentiment to determine if the review is positive or \
    negative

@author: Alfiqmal
"""

import pandas as pd
from sentiment_analysis_modules import ExploratoryDataAnalysis, ModelCreation, ModelEvaluation
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

URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), "model.h5")
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), "tokenizer_data.json")
# EDA

# STEP 1: Import Data

df = pd.read_csv(URL)
review = df["review"]
sentiment = df["sentiment"]

# STEP 2: Data Cleaning

# Remove tags

eda = ExploratoryDataAnalysis()
review = eda.remove_tags(review)
review = eda.lower_split(review)

# STEP 3: Features Selection

# STEP 4: Data Vectorization

review = eda.sentiment_tokenizer(review, TOKENIZER_JSON_PATH)
review = eda.sentiment_pad_sequence(review)

# STEP 5: Data Preprocessing

one_hot_encoder = OneHotEncoder(sparse = False)
sentiment = one_hot_encoder.fit_transform(np.expand_dims(sentiment, axis = -1))

# to calculate number of total categories

nb_categories = len(np.unique(sentiment))

# X = review, y = sentiment
# Train test split

X_train, X_test, y_train, y_test = train_test_split(review,
                                                    sentiment,
                                                    test_size = 0.3,
                                                    random_state = 123)

X_train = np.expand_dims(X_train, axis = -1)
x_test = np.expand_dims(X_test, axis = -1)

print(y_train[0])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis = 0)))

#%% Model Creation

mc = ModelCreation()

num_words = 10000

model = mc.LSTM_layer(num_words, nb_categories)
log_files = os.path.join(LOG_PATH,
                         datetime.datetime.now().strftime("%Y%m%d - %H%M%S"))

tensorboard_callback = TensorBoard(log_dir = log_files, histogram_freq = 1)

#%% compile and model fitting

model.compile(optimizer = "adam",
              loss = "categorical_crossentropy",
              metrics = "acc")

model.fit(X_train, y_train,
          epochs = 5,
          validation_data = (X_test, y_test),
          callbacks = tensorboard_callback)

#%% Model Evaluation

predicted_advanced = np.empty([len(X_test), 2])

for index, test in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test, axis = 0))
    
#%% Model Analysis

y_pred = np.argmax(predicted_advanced, axis = 1)
y_true = np.argmax(y_test, axis = 1)

me = ModelEvaluation()
me.report_metrics(y_true, y_pred)

#%% Model Deployment

model.save(MODEL_SAVE_PATH)
