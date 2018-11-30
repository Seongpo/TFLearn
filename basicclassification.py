'''This code is from https://www.tensorflow.org/tutorials/keras/basic_classification '''

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
                'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(train_images.shape)
# print(len(train_labels))

# plt.figure()
# plt.imshow(train_images[100])
# plt.colorbar()
# plt.grid(False)
# plt.show()

'''We scale these values to a range of 0 to 1 before feeding to the neural network model. '''

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize = (10,10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap = plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

''' Build the model 
Most of deep learning consists of chaining together simple layers. Most layers, like
  * tf.keras.layers.Dense, have parameters that are learned during training.
  * tf.keras.layers.Flatten, transforms the format of the images from a 2d-array (of 28 by 28 pixels),
 to a 1d-array of 28 * 28 = 784 pixels
'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
])

'''Compile the model
Before the model is ready for training, it needs a few more settings. 
 * Loss function —This measures how accurate the model is during training. 
  We want to minimize this function to "steer" the model in the right direction.

 * Optimizer —This is how the model is updated based on the data it sees and its loss function.

 * Metrics —Used to monitor the training and testing steps. 
 The following example uses accuracy, the fraction of the images that are correctly classified.
'''

model.compile(optimizer = tf.train.AdamOptimizer(), 
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)

'''Make predictions'''

predictions = model.predict(test_images)

# print(predictions[0], 
# '예측된 분류 {}, 와 원분류 {} 의 비교'.format(np.argmax(predictions[0]), test_labels[0]))

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color = color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = '#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions, test_labels)
# plt.show()

# num_rows = 5
# num_cols = 4
# num_images = num_rows*num_cols
# plt.figure(figsize = (2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, predictions, test_labels)
# plt.show()

'''traned model to make a prediction'''
img = test_images[0]  #img.shape == (28,28)
img = (np.expand_dims(img, 0))  # expanded_img.shape ==(1,28,28)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_=plt.xticks(range(10), class_names, rotation = 45)
plt.show()