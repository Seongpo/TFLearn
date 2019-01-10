from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28)/255.0
test_images = test_images[:1000].reshape(-1, 28*28)/255.0

# Return a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer = tf.train.AdamOptimizer(),
                loss = tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
    return model

model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy : {:5.2f}%".format(100*acc))

checkpoint_path = "./training_2/cp-{epoch:04d}.ckpt"
# model.load_weights(checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)
#latest = tf.train.latest_checkpoint(checkpoint_dir)
#print(latest)
# model = create_model()
model.load_weights("C:/Users/spbae/OneDrive/Docs/Python/Tensorflow/training_2/cp-0050.ckpt")
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy : {:5.2f}%".format(100*acc))
