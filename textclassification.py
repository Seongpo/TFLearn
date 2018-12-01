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
# 딕셔너리 타입의 단어와 자연수를 매핑 시킨 값을 리턴받아서
word_index = imdb.get_word_index()

# 0,1,2,3 은 특정단어로 남김. 따라서 모든 단어에 3씩 값을 더해줌. 
# dictionary.items() 는 딕셔너리의 key, value값을 튜플형태로 돌려줌
#The first indices are reserved
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# 딕셔너리의 key와 value 값을 뒤집어서 새로운 딕셔너리를 만듦
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 텍스트 값을 받아서 인덱스의 value 값(integer)을 받아서 스트링으로 변환
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_review(train_data[0]))
# train_data와  test_data를 keras 를 이용하여 256 길이로 시퀀스 뒤에('post') key 값 '<PAD>' 값인 0으로 
# 채워 tensor로 변환
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
=====BUILD the Model=====
The neural network is created by stacking layers—this requires two main architectural decisions:

How many layers to use in the model?
How many hidden units to use for each layer?

임베딩 레이어는 정수로 인코딩된 단어를 취해서 각 워드 인덱싱에 대한 임베딩 벡터를 찾는다. 이 벡터들이 모델 트레인으로
러닝 된다.  벡터들은 출력 배열에 대한 차원을 더한다. 
GlobalAveragePooling1D 레이어는 각 예제에 대해 고정길이의 출력 벡터를 리턴한다. 이것이 입력변수길이를 다룰수 있도록 한다.
고정길이의 출력은 16개의 히든 레이어를 가지는 fully-connected(Dense) 레이어에 연결된다.
마지막 레이어는 싱글 출력 노드로 연결된다. sigmoid 함수이고 출력은 0과 1 사이의 부동소수 이다. 
'''

# input shape is the vocabulary count used for the movie reviews(10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())

'''
loss function and optimizer
loss funciton(cost function) 과 최소값을 찾아가는 방법이 필요함
여기서는 binary_crossentropy 를 loss funtion으로 AdamOptimizer()를 옵디마이져로 사용함

'''
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


'''
=== Create a validation set ===
트레이닝 데이터와 별도로 10000 의 Validation 세트를 마련함
'''
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

'''
=== Train the model ===
40 epoch와 512 개의 batch 사이즈를 가지는 트레이닝 모델을 만들어서 학습
미리 만들어둔 10000 개의 validation 데이터를 이용해서 매 epoch 마다 평가
'''

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 평가 함수를 통해 트레이닝 된 결과를 평가함
results = model.evaluate(test_data, test_labels)
print(results)
print(history.history)
'''
아래 트레이닝 결과에 대한 그래프
'''
# model.fit() returns a History object 
# that contains a dictionary with everything that happened during training:
# history.history 는 딕셔너리로서 'val_loss', 'val_acc', 'loss', 'acc' 를 key 로 가진다.
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
