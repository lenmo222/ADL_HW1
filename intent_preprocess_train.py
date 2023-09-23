import os 
import json
import keras
import pickle
import argparse
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
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--eval_file", type=str, default="")
    return parser.parse_args()
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

train_text = []
train_intent = []
val_text = []
val_intent = []

for i in train:
    train_text.append(i['text'])
    train_intent.append(i['intent'])
for i in val:
    val_text.append(i['text'])
    val_intent.append(i['intent'])
# train text text_vector
layer_text = layers.TextVectorization(max_tokens=20000, output_sequence_length=30)
layer_text.adapt(train_text)
vectorized_text = layer_text(train_text)

# train intent text_vector
layer_intent = layers.TextVectorization()
layer_intent.adapt(train_intent)
vectorized_intent = layer_intent(train_intent)

# build a dict on train intent
intent_set = set()
for i in train_intent:
    intent_set.add(i)
intent_set_dict = dict(zip(intent_set,range(len(intent_set))))
inent_label_list = []
for i in train_intent:
    inent_label_list.append(intent_set_dict[i])

saveDictionary(intent_set_dict,'./intent_set_dict.pkl')
# val text text_vector
val_vectorized_text = layer_text(val_text)
# val intent text_intent 
val_vectorized_intent = layer_intent(val_intent)

# val intent dict id on train_intent
val_inent_label_list = []
for i in val_intent:
    val_inent_label_list.append(intent_set_dict[i])

voc = layer_text.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

# Pickle the config and weights
pickle.dump({'config': layer_intent.get_config(),
             'weights': layer_intent.get_weights()}
            , open("./intent_intent_layer.pkl", "wb"))
# Pickle the config and weights
pickle.dump({'config': layer_text.get_config(),
             'weights': layer_text.get_weights()}
            , open("./intent_text_layer.pkl", "wb"))

num_tokens = len(voc) + 2
embedding_dim = 300
hits = 0
misses = 0
# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None and embedding_vector.size > 0:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

train_set = tf.data.Dataset.from_tensor_slices(
        (vectorized_text, inent_label_list)).batch(128)
val_set = tf.data.Dataset.from_tensor_slices(
        (val_vectorized_text, val_inent_label_list)).batch(128)


#train
vocab_length = 10000
model = keras.Sequential()

model.add(layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
))


model.add(layers.Bidirectional(layers.LSTM(128,return_sequences=True,dropout=0.3)))
model.add(layers.Bidirectional(layers.LSTM(128,dropout=0.2)))
model.add(layers.Dense(150, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"],
)
model.summary()

model.fit(
   train_set, batch_size=128, epochs=20,validation_data=val_set
)

model.save('ADL_HW1_intent_model.h5')
