from tensorflow.keras import layers     # class to create the Dense deep net
from tensorflow import keras            # Interface of the tensorflow NN libraries
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


# >> Creating the Neural Net model
def build_and_compile_model(norm, out):
    # creating the densly connected deep conv neural network
    model = keras.Sequential([
        norm,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
        out
    ])

    # compiling the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    return model


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


column_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'edge']

# skiprows=skip to use skip
raw_dataset = pd.read_csv('testing.csv', names=column_names)
dataset = raw_dataset.copy()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('edge')
test_labels = test_features.pop('edge')


p_normalizer = layers.Normalization(input_shape=[8, ], axis=None)
p_normalizer.adapt(train_features)
o_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
o_normalizer.adapt(train_labels)

# >> creating and Training the model
p_model = build_and_compile_model(p_normalizer, o_normalizer)
history = p_model.fit(
    train_features,  # the inputs to  the model
    train_labels,   # the labels we want the model to train for
    epochs=10000,   # epoch count
    verbose=2,      # Log Progress
    # Calculate validation results on 20% of the training data.
    validation_split=0.2,
    batch_size=200)


# >> saving the model
p_model.save('dnn_model_classi')
