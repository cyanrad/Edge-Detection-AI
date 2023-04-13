# class to create the Dense deep net
from tensorflow.keras import layers, models
# Interface of the tensorflow NN libraries
from tensorflow import keras
import tensorflow as tf                 # ML and AI library
import numpy as np                      # arrays and matrices
import pandas as pd                     # to read the training data
import random                           # to acquire data from the data set randomly

# >> disabling the graphics card
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# checking which device is used to compute the neural net
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

n = 44836  # number of records in file
s = 40000  # desired sample size
column_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'edge']
# used for getting random data from the set
# use this when we don't want the entire data set
skip = sorted(random.sample(range(n), n-s))

# skiprows=skip to use skip
column_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'edge']
raw_dataset = pd.read_csv('testing.csv', names=column_names, skiprows=skip)
dataset = raw_dataset.copy()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('edge')
test_labels = test_features.pop('edge')

np_train_features = train_features.to_numpy()
np_train_features = np_train_features.reshape([-1, 4, 2])
np_test_features = test_features.to_numpy()
np_test_features = np_test_features.reshape([-1, 4, 2])


p_normalizer = layers.Normalization(input_shape=[4, 2], axis=None)
p_normalizer.adapt(train_features)
o_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
o_normalizer.adapt(train_labels)


model = models.Sequential()
model.add(layers.Conv1D(32, 8, activation='relu', input_shape=[4, 2]))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv1D(64, 8, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv1D(64, 8, activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.add(o_normalizer)


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])


history = model.fit(np_train_features, train_labels, epochs=10, verbose=2,
                    validation_data=(np_test_features, test_labels))

history.save('dnn_model_32_con')
