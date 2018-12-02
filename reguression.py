from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

# print(tf.__version__)

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# print("Tranining set: {}".format(train_data.shape))
# print("Testing set: {}".format(test_data.shape))

'''
1. Per capita crime rate.(1인당 범죄율)
2. The proportion of residential land zoned for lots over 25,000 square feet.
(25,000 평방 피트가 넘는 지역에 거주하는 토지의 비율)
3. The proportion of non-retail business acres per town.
(마을 당 비 소매 사업 에이커의 비율)
4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
찰스 리버 더미 변수(강의 경계에 있으면 1, 그렇지 않으면 0)
5. Nitric oxides concentration (parts per 10 million).
질소 산화물 농도(1000만 명당)
6. The average number of rooms per dwelling.
주거당 평균 방수
7. The proportion of owner-occupied units built before 1940.
1940년 이전에 건설된 소유 점령 단위의 비율
8. Weighted distances to five Boston employment centers.
5개의 보스턴 고용 센터에 가중치 적용
9. Index of accessibility to radial highways.
방사형 고속도로 접근성 지수
10. Full-value property-tax rate per $10,000.
$10,000 당 부당한 재산세 비율
11. Pupil-teacher ratio by town.
마을 별 학생-교사 비율
12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
1000*(Bk - 0.63)**2
13. Percentage lower status of the population.
인구의 낮은 지위의 백분율
'''
print(train_data[0])

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns = column_names)
print(df.head())
print(train_labels[0:10])

# Test data is *no* used when calulating the mean and std
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0]) # First training sample, normalized

'''
=== Create the model ===
Sequential 모델
2개의 densely connected 된 hidden layer에 1개의 출력을 가지는 모델
'''
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation = tf.nn.relu,
                            input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation = tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mae'])
    return model
model = build_model()
model.summary()

'''
Train the model
'''
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('???')
        print('.', end = '')
EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

import matplotlib.pyplot as plt
print(history.history.keys())

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])
  plt.show()

# plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
plt.show()