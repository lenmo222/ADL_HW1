import os 
import json
import keras
import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import array
from tensorflow import keras
from keras.utils import np_utils
from tensorflow.keras import layers
from keras.layers.core import Dropout
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Dense, Embedding

def parser():
    parser = argparse.ArgumentParser(description="ADL HW1 slot")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--pred_file", type=str, default="")
    return parser.parse_args()
def loadDictionary(file):
  import pickle
  a_file = open(file, "rb")
  dt = pickle.load(a_file)
  return dt
words_dict = loadDictionary("./slot_words_dict.pkl")
tags_dict = loadDictionary("slot_tags_dict.pkl")
def get_tokens_value(token):
    for key,value in words_dict:
        if(key==token):
            return value
    return 0
def get_tags_value(token):
    for key,value in tags_dict:
        if(key==token):
            return value
    return 0

args = parser()
if args.test_file:
    with open(args.test_file) as f:
        test = json.load(f)
test_text = []
test_id = []
for i in test:
    test_text.append(i['tokens'])
    test_id.append(i['id'])

test_vector_tokens = []
test_id = []
for i in test:
    test_tokens = i["tokens"]
    test_vector_token = []
    test_id.append(i["id"])
    for j in test_tokens:
        key = get_tokens_value(j)
        test_vector_token.append(key)
    test_vector_tokens.append(test_vector_token)
for i in test_vector_tokens:
    time = 35-len(i)
    for j in range(0,time):
        i.append(0)
word_index = []
for i in words_dict:
    word_index.append(i[0])

model = tf.keras.models.load_model("./ADL_HW1_slot_model.h5")
predict = model.predict(test_vector_tokens)
predict_reduction = []
for i in range(len(predict)):
    predict_class_list = []
    s = ""
    for j in range(len(test[i]['tokens'])):
        index_max = np.argmax(predict[i][j])
        if(index_max==0):
            s += 'O '
        else:
            s += tags_dict[np.argmax(predict[i][j])][0]
            s += ' '
    if(len(s)==0):
        len_s = len(test[i]['tokens'])
        for j in range(len_s):
            s += 'O'
    if(len(s)!=1):
        s = s[:-1]
    predict_reduction.append(s)
if args.pred_file:
    df = pd.DataFrame(predict_reduction, index=test_id, columns=['tags'])
    df.index.name = 'id'
    df.to_csv(args.pred_file+".csv")


