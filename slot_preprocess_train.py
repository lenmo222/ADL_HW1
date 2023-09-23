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
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--eval_file", type=str, default="")
    return parser.parse_args()
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
def saveDictionary(dt, file):
    import pickle
    a_file = open(file, "wb")
    pickle.dump(dt, a_file)
    a_file.close()
args = parser()
if args.train_file:
    with open(args.train_file) as f:
        train = json.load(f)
if args.eval_file:
    with open(args.eval_file) as f:
        val = json.load(f)

path_to_glove_file = os.path.join(
    os.path.expanduser("~"), "/glove.840B.300d.txt"
)
embeddings_index = {}
with open('./glove.840B.300d.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))
train_word_list = []
train_tags = []
words_set = set()
tags_set = set()
for i in train:
    train_word = i["tokens"]
    train_tag = i["tags"]
    for j in train_word:
        train_word_list.append(j)
        words_set.add(j)
    for j in train_tag:
        train_tags.append(j)
        tags_set.add(j)
for i in val:
    val_word = i["tokens"]
    for j in val_word:
        words_set.add(j)

words_dict = list(zip(words_set, range(len(words_set))))
tags_dict =  list(zip(tags_set, range(len(tags_set))))

saveDictionary(words_dict,"./slot_words_dict.pkl")
saveDictionary(tags_dict,"./slot_tags_dict.pkl")

vector_tokens = []
vector_tags = []
max_len = 0


for i in train:
    train_tokens = i["tokens"]
    vector_token = []
    
    train_tags = i["tags"]
    vector_tag = []
    for j in train_tokens:
        key = get_tokens_value(j)
        vector_token.append(key)
    max_len = max(max_len,len(vector_token))
    vector_tokens.append(vector_token)
    for j in train_tags:
        key = get_tags_value(j)
        vector_tag.append(key)
    vector_tags.append(vector_tag)

    
val_vector_tokens = []
val_vector_tags = []
for i in val:
    val_tokens = i["tokens"]
    val_vector_token = []
    
    val_tags = i["tags"]
    val_vector_tag = []
    for j in val_tokens:
        key = get_tokens_value(j)
        val_vector_token.append(key)
    val_vector_tokens.append(val_vector_token)
    for j in val_tags:
        key = get_tags_value(j)
        val_vector_tag.append(key)
    val_vector_tags.append(val_vector_tag)
    
for i in vector_tokens:
    time = max_len-len(i)
    for j in range(0,time):
        i.append(0)
for i in vector_tags:
    time = max_len-len(i)
    for j in range(0,time):
        i.append(0)
for i in val_vector_tokens:
    time = max_len-len(i)
    for j in range(0,time):
        i.append(0)
for i in val_vector_tags:
    time = max_len-len(i)
    for j in range(0,time):
        i.append(0)

word_index = []
for i in words_dict:
    word_index.append(i[0])

num_tokens = len(word_index)
embedding_dim = 300
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for i in range(1,len(word_index)):
    embedding_vector = embeddings_index.get(word_index[i])
    if embedding_vector is not None and embedding_vector.size > 0:
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))  

train_set = tf.data.Dataset.from_tensor_slices(
        (vector_tokens, vector_tags)).batch(128)
val_set = tf.data.Dataset.from_tensor_slices(
        (val_vector_tokens, val_vector_tags)).batch(128)


vocab_length = 20000
model = keras.Sequential()

# model.add(layers.Embedding(vocab_length,100))
model.add(layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
#     trainable=False,
))
model.add(layers.Bidirectional(layers.LSTM(128,dropout=0.3,return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(128,dropout=0.25,return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(128,dropout=0.15,return_sequences=True)))
model.add(layers.TimeDistributed(layers.Dense(9)))
model.add(layers.Dense(max_len, activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"],
)
model.summary()
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor = 'val_accuracy'),
]
model.fit(
   train_set, batch_size=128, epochs=20,validation_data=val_set,callbacks=my_callbacks
)
model.save('ADL_HW1_slot_model.h5')