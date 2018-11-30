'''This code is from https://www.tensorflow.org/tutorials/keras/basic_text_classification 

This notebook classifies movie reviews as positive or negative using the text of the review. 
This is an example of binary—or two-class—classification, 
an important and widely applicable kind of machine learning problem
'''

import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# print("Training entries: {}, labels: {}".format(len(train_data), len(test_labels)))
# print(train_data[0])
# print(len(train_data[0]), len(train_data[1]))

# Convert the integers back to words
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

#The first indices are reserved
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data, 
                                                        value=word_index["<PAD>"], 
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["<PAD>"],
                                                        padding = 'post',
                                                        maxlen = 256)
# print(train_data[0], len(train_data[0]))
# print(decode_review(train_data[0]))

'''
The neural network is created by stacking layers—this requires two main architectural decisions:

How many layers to use in the model?
How many hidden units to use for each layer?
'''

# input shape is the vocabulary count used for the movie reviews(10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

history_dict = history.history
history_dict.keys()
print(dict_keys(['val_acc', 'acc', 'val_loss', 'loss']))