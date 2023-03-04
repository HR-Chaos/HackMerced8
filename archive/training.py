import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, SubsetRandomSampler, DataLoader
from model import LSTMmodel
from datetime import datetime

def datestring_to_unixtimestamp(date_string):
    date = datetime.strptime(date_string, '%Y-%m-%d')
    unix_timestamp = date.toordinal() - datetime(1950, 1, 1).toordinal()  # convert to days since 1950-01-01
    return unix_timestamp
####################################################################################################################
# Date loading and preprocessing

# Load the CSV file into a pandas dataframe
df = pd.read_csv('Weather_data_Fresno_cleaned.csv')

# Convert the dates into Unix timestamps
date_column = df['DATE']
unix_timestamps = [datestring_to_unixtimestamp(date_string) for date_string in date_column]
tensor_unix_timestamps = torch.tensor(unix_timestamps)

dates = pd.to_datetime(df['DATE'])
timestamps = (dates - pd.Timestamp("1950-01-01")) // pd.Timedelta('1 day')

####################################################################################################################
# Convert the timestamps to a numpy array and then to a PyTorch tensor

tensor_timestamps = torch.tensor(unix_timestamps).float()
print(tensor_timestamps)
tensor_temp_max = torch.from_numpy(np.array(df['TMAX'])).float()
print(tensor_temp_max)
tensor_temp_min = torch.from_numpy(np.array(df['TMIN'])).float()
print(tensor_temp_min)
tensor_precipitation = torch.from_numpy(np.array(df['PRCP'])).float()
print(tensor_precipitation)

# Print the shape of the tensor
print("time_stamps in days:", tensor_timestamps.shape)
print("temp_max in degrees F:", tensor_temp_max.shape)
print("temp_min in degrees F:", tensor_temp_min.shape)
print("precipitation in inches:", tensor_precipitation.shape)


####################################################################################################################
# Combine the input and target tensors into a dataset
dataset = TensorDataset(tensor_timestamps, tensor_temp_min, tensor_temp_max, tensor_precipitation)

# Split the dataset into training, validation, and testing sets
num_samples = len(dataset)
indices = list(range(num_samples))
np.random.shuffle(indices)

# split into 60% training, 20% validation, and 20% testing
split1 = int(num_samples * 0.6)
split2 = int(num_samples * 0.8)
train_indices = indices[:split1]
val_indices = indices[split1:split2]
test_indices = indices[split2:]

# Create the samplers
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

####################################################################################################################
# Training the model

# Step 0: Define hyperparameters
num_epochs = 100
input_size = 1
hidden_size = 32
output_size = 3
learning_rate = 0.001
batch_size = 32

# Step 1: Create DataLoader objects for the training, validation, and testing sets
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# Step 2: Define model architecture
model = LSTMmodel(input_size, hidden_size, output_size)

# Step 3: Define loss function
criterion = nn.MSELoss()

# Step 4: Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Step 5: Train the model
print("Training the model...")
for epoch in range(num_epochs):
    for dates, min_temps, max_temps, precipitations in train_loader:
        optimizer.zero_grad()
        input_tensor = torch.Tensor([timestamp for timestamp in dates]).unsqueeze(1)
        input_tensor = input_tensor.view(input_tensor.shape[0], 1, -1)
        output_tensor = model(input_tensor)
        target_tensor = torch.stack([min_temps, max_temps, precipitations], dim=1)
        loss = criterion(output_tensor, target_tensor)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} loss: {loss.item()}")

print("Training complete!")
# Step 6: Evaluate the model on the validation and testing sets
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        output = model(x)
        target = torch.stack([y[:, 0], y[:, 1], y[:, 2]], dim=1)
        loss = criterion(output, target)
    for x, y in test_loader:
        output = model(x)
        target = torch.stack([y[:, 0], y[:, 1], y[:, 2]], dim=1)
        loss = criterion(output, target)