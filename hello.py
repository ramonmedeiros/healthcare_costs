import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

# read dataset
dataset = pd.read_csv('insurance.csv')

# map gender, smoker
dataset['smoker'] = dataset['smoker'].map({0: "no", 1: 'yes'})
dataset['sex'] = dataset['sex'].map({0: "female", 1: 'male'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

# split data
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# info about dataset
print(train_dataset.describe().transpose())

# get features and labels 
train_features = train_dataset.copy()
train_labels = train_features.pop("expenses")
test_features = test_dataset.copy()
test_labels = test_features.pop("expenses")

# using multiple inputs
normalizer = layers.experimental.preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

# linear regression model
model = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error',
    metrics=['mae', 'mse'])

model.summary()

history = model.fit(x=train_features, y=train_labels,
                    epochs=100,
                    verbose=0,
                    validation_split = 0.2)  # Calculate validation results on 20% of the training data

# parse results
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plot_loss(history)

test_dataset = test_features

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
plt.show()
