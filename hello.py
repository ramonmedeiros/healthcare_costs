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
  plt.ylabel('Error [Expenses]')
  plt.legend()
  plt.grid(True)
  plt.show()

# read dataset
dataset = pd.read_csv('insurance.csv')

# map gender, smoker
dataset['smoker'] = dataset['smoker'].map({"no": 0, 'yes': 1})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

print(dataset.head())

# split data
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# remove expenses
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# normalize data
normalizer = layers.experimental.preprocessing.Normalization()
normalizer.adapt(np.array(train_dataset))

# linear regression model
model = keras.Sequential([
      normalizer,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss=tf.losses.MeanAbsoluteError(),
    metrics=[tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredError()])

model.summary()

history = model.fit(x=train_dataset, y=train_labels,
                    epochs=300,
                    verbose=0,
                    validation_split = 0.2)  # Calculate validation results on 20% of the training data

# plot loss
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plot_loss(history)

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
