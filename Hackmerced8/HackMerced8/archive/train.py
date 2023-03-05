import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import weatherModel

# Load data
data = pd.read_csv("Weather_data_Fresno_cleaned.csv")

# Convert date strings to float values
dates = pd.to_datetime(data["DATE"])
data["DATE"] = (dates - dates.min()) / np.timedelta64(1, "D")
print(dates.min())

# Normalize data
mean = data.mean()
std = data.std()
data = (data - mean) / std

# Split data into training and testing sets
split_ratio = 0.8
train_size = int(len(data) * split_ratio)
train_data, test_data = data[:train_size], data[train_size:]

# Convert data to PyTorch tensors
train_x = torch.from_numpy(train_data["DATE"].values.astype(np.float32))
train_y = torch.from_numpy(train_data[["TMIN", "TMAX", "PRCP"]].values.astype(np.float32))
test_x = torch.from_numpy(test_data["DATE"].values.astype(np.float32))
test_y = torch.from_numpy(test_data[["TMIN", "TMAX", "PRCP"]].values.astype(np.float32))
print("train_x: ", train_x)
print("train_y: ", train_y)

# Create PyTorch dataset and dataloader objects
train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define loss function and optimizer
model = weatherModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train model
num_epochs = 10

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in test_loader:
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, targets)
            total_loss += loss.item() * len(inputs)
        avg_loss = total_loss / len(test_data)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
# Save model
torch.save(model.state_dict(), "m1.pt")

# Create new instance of the model
model = weatherModel()

# Load trained parameters
model.load_state_dict(torch.load("m1.pt"))

# Prepare input data for prediction
input_date = pd.to_datetime("2021-03-04")
input_date = (input_date - dates.min()) / np.timedelta64(1, "D")
input_x = torch.tensor(input_date, dtype=torch.float32).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(input_x)
    output = output.numpy()[0]

# Convert prediction to original units
output = output * std[["TMIN", "TMAX", "PRCP"]].values + mean[["TMIN", "TMAX", "PRCP"]].values

# Print prediction
print(f"Predicted min temp: {output[0]:.2f}")
print(f"Predicted max temp: {output[1]:.2f}")
print(f"Predicted precipitation: {output[2]:.2f}")


