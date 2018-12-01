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
<<<<<<< HEAD
# 딕셔너리 타입의 단어와 자연수를 매핑 시킨 값을 리턴받아서
word_index = imdb.get_word_index()

# 0,1,2,3 은 특정단어로 남김. 따라서 모든 단어에 3씩 값을 더해줌. 
# dictionary.items() 는 딕셔너리의 key, value값을 튜플형태로 돌려줌
=======
word_index = imdb.get_word_index()

>>>>>>> 792b44c5b978e94b492a0e16dff41e0d99aa2b95
#The first indices are reserved
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

<<<<<<< HEAD
# 딕셔너리의 key와 value 값을 뒤집어서 새로운 딕셔너리를 만듦
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 텍스트 값을 받아서 인덱스의 value 값(integer)을 받아서 스트링으로 변환
=======
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

>>>>>>> 792b44c5b978e94b492a0e16dff41e0d99aa2b95
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_review(train_data[0]))

<<<<<<< HEAD
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
=======
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
                                                       
print(len(train_data[0]), len(test_data[1]))
# print(train_data[0])

# input shape is the vocabulary count used for the movie reviews (10,000 words)
>>>>>>> 792b44c5b978e94b492a0e16dff41e0d99aa2b95
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

<<<<<<< HEAD
print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
=======
# print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])
>>>>>>> 792b44c5b978e94b492a0e16dff41e0d99aa2b95

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
<<<<<<< HEAD
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
=======
                    verbose = 1)

results = model.evaluate(test_data, test_labels)
print(results)


>>>>>>> 792b44c5b978e94b492a0e16dff41e0d99aa2b95
