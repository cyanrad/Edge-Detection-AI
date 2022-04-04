from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt  # math plots
import numpy as np
import pandas as pd
import random

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

n = 44836  # number of records in file
s = 4000  # desired sample size
# , skiprows=skip
column_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'edge']
skip = sorted(random.sample(range(n), n-s))
raw_dataset = pd.read_csv('testing.csv', names=column_names, skiprows=skip)
dataset = raw_dataset.copy()


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('edge')
test_labels = test_features.pop('edge')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))


p_normalizer = layers.Normalization(input_shape=[8, ], axis=None)
p_normalizer.adapt(train_features)
o_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
o_normalizer.adapt(train_labels)

p_model = tf.keras.Sequential([
    p_normalizer,
    layers.Dense(units=9, activation='relu'),
    o_normalizer
])

p_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = p_model.fit(
    train_features,
    train_labels,
    epochs=1000,
    # Suppress logging.
    verbose=2,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)
p_model.save('lin_model_1k_6k_o')
test_predictions = p_model.predict(
    np.array([102, 100, 106, 104, 104, 113, 108, 101]))


plt.show()
