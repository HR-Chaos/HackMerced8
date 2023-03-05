import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv('train_data.csv')

# Select relevant features and target variable

# temperature in F and precipitation in inches
features = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 
            'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12',]
target = 'yield'
x = df[features].values
y = df[target].values

print(x)
print(y)

# Split data into training and testing sets
train_size = int(len(x) * 0.8)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Normalize the input data
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Reshape input data to match expected shape of LSTM layer
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))

# Load the saved model from disk
loaded_model = tf.keras.models.load_model('my_model.h5')

# Evaluate the loaded model on test data
test_loss = loaded_model.evaluate(x_test, y_test)
print(test_loss)

# Define the input sequence for a new year
# 2021 was removed from data
new_year = np.array([[50.3,53.4,53.7,62.8,69.1,75.3,79.8,77.6,74.7,62.7,56.5,46,3.06,0.92,0.98,0.07,0,0,0,0,0.02,3.96,0.56,4.65]])
x = new_year[:, :24].reshape(-1, 24)

# Normalize the input data using the mean and standard deviation from the training data
new_year = (new_year - mean) / std

# Generate a prediction for the yield based on the input sequence
predictions = loaded_model.predict(x)
# print('mean:', predictions[0][0]*mean[-1])
# print('std:', predictions[0][0]*std[-1])
#predictions = (predictions * std[-1]) + mean[-1]

# Extract the predicted yield from the output sequence
print('Predicted yield for 2023:', predictions[0][0])