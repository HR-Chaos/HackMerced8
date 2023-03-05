import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

print('\n\n-----------------------------------------------------------------------------------------------\n\n')
# Load data

# Load the dataset into a pandas dataframe
df = pd.read_csv('train_data.csv')

#randomize
df = df.sample(frac=1).reset_index(drop=True)

# Select relevant features and target variable

# temperature in F and precipitation in inches
features = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 
            'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12']
target = 'yield'
x = df[features].values
y = df[target].values

print(x)
print(y)

# Split data into training and testing sets
train_size = int(len(x) * 0.9)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Normalize the input data
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Reshape input data to match expected shape of LSTM layer
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))


# Define the RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(24,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Define the learning rate scheduler
def lr_scheduler(epoch):
    lr = 0.1
    if epoch >200:
        lr *= 0.1
    return lr
# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opt, loss='mse')
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# Train the model
history = model.fit(x_train, y_train, epochs=300, batch_size=16, validation_data=(x_test, y_test), callbacks=[lr_callback])
# Evaluate the model on test data
test_loss = model.evaluate(x_test, y_test)
# Save the model to disk
model.save('my_model.h5')

print('\n\n-----------------------------------------------------------------------------------------------\n\n')

# Load the saved model from disk
loaded_model = tf.keras.models.load_model('my_model.h5')

# Evaluate the loaded model on test data
test_loss = loaded_model.evaluate(x_test, y_test)

# Plot the training and validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()