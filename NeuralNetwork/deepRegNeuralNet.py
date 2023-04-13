from tensorflow.keras import layers     # class to create the Dense deep net
from tensorflow import keras            # Interface of the tensorflow NN libraries
import tensorflow as tf                 # ML and AI library
import numpy as np                      # arrays and matrices
import pandas as pd                     # to read the training data

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
    # >> creating the densly connected deep regression neural network
    # using relu activation function (rectified linear unit)
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        out
    ])

    # >> compiling the model
    # error function Computes the mean of absolute difference between labels and predictions.
    # loss = abs(y_true - y_pred)
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# names of the columns for extracting data using pandas and organizing it
column_names = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'edge']

# reading the data from the CSV
dataset = pd.read_csv('data_hand_drawn.csv', names=column_names)
# randomizing the data set
dataset = dataset.sample(frac=1, random_state=1).reset_index()

# >> splitting the data between the training and testing with a ratio of 80/20
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# creating the train features data frame, inputs to the neural network
train_features = train_dataset.copy()
test_features = test_dataset.copy()

# creating the train labels which are the output's ground truth
train_labels = train_features.pop('edge')
test_labels = test_features.pop('edge')

# making sure that there is no index variable
train_features.pop('index')
test_features.pop('index')

# >> printing the first five elements in the train features to
# make sure the data we have is in the correct form
print(train_features.head())

# >> creating input and output layers
p_normalizer = layers.Normalization(input_shape=[8, ], axis=None)
p_normalizer.adapt(train_features)
o_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
o_normalizer.adapt(train_labels)

# >> creating the model
p_model = build_and_compile_model(p_normalizer, o_normalizer)

# >> training the model
history = p_model.fit(
    train_features,  # the inputs to  the model
    train_labels,    # the labels we want the model to train for
    epochs=100,      # epoch count
    verbose=2,       # Log Progress
    # Calculate validation results on 20% of the training data.
    validation_split=0.2,
    batch_size=400)


# >> saving the model
p_model.save('hand_1K')
