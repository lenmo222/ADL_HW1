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
    parser = argparse.ArgumentParser(description="ADL HW1 intent")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--pred_file", type=str, default="")
    return parser.parse_args()
def loadDictionary(file):
  import pickle
  a_file = open(file, "rb")
  dt = pickle.load(a_file)
  return dt
intent_set_dict = loadDictionary("./intent_set_dict.pkl")
def get_key(val):
    for key, value in intent_set_dict.items():
         if val == value:
             return key
args = parser()

if args.test_file:
    with open(args.test_file) as f:
        test = json.load(f)

test_text = []
test_id = []
for i in test:
    test_text.append(i['text'])
    test_id.append(i['id'])

layer_text_load = pickle.load(open("./intent_text_layer.pkl", "rb"))
layer_text = TextVectorization.from_config(layer_text_load['config'])
layer_text.adapt(test_text)
layer_text.set_weights(layer_text_load['weights'])


test_vectorized_text = layer_text(test_text)

model = tf.keras.models.load_model("./ADL_HW1_intent_model.h5")

predict = model.predict(test_vectorized_text)

predict_class = []
predict_reduction = []
for i in predict:
    predict_class.append(np.argmax(i))
for i in predict_class:
    predict_reduction.append(get_key(i))
if args.pred_file:
    df = pd.DataFrame(predict_reduction, index=test_id, columns=['intent'])
    df.index.name = 'id'
    df.to_csv(args.pred_file+".csv")




