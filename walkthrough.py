'''Setup program'''

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf

tf.enable_eager_execution()

# print("TensorFlow version: {}".format(tf.__version__))
# print("Eager execution: {}".format(tf.executing_eagerly()))

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
# This returns the file path of the downloaded file.
train_dataset_fp = tf.keras.utils.get_file(fname = os.path.basename(train_dataset_url), origin = train_dataset_url)

# print("Local copy of the dataset file: {}".format(train_dataset_fp))

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

# Create a tf.data.Dataset

batch_size = 32
# batch_size = 100

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names = column_names,
    label_name = label_name,
    num_epochs = 1)

features, labels = next(iter(train_dataset))

# plt.scatter(features['petal_length'],
#             features['sepal_length'],
#             c = labels,
#             cmap = 'viridis')
# plt.xlabel("Petal length")
# plt.ylabel("Sepal length")
# plt.show()

def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis = 1)
    return features, labels

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))
print(features[:5])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)), 
    tf.keras.layers.Dense(10, activation=tf.nn.relu), 
    tf.keras.layers.Dense(3)
    ])

"""USING THE MODEL
predictions = model(features)
predictions[:5]
tf.nn.softmax(predictions[:5])
print("Prediction: {}".format(tf.argmax(predictions, axis = 1)))
print("    Labels: {}".format(labels))
"""

def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels = y, logits = y_)

l = loss(model, features, labels)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
global_step = tf.Variable(0)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(global_step.numpy(), loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.variables), global_step)

print("Step: {},         Loss: {}".format(global_step.numpy(), loss(model, features, labels).numpy()))

##Note: Rerunning this cell uses the same model variables

#keep results for plotting

train_loss_results = []
train_accuracy_results = []