import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import Model

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

model = Model()
model.load_state_dict(torch.load('model.pt'))
# Switch model to evaluation mode
model.eval()

# Preprocess the input
date_str = '2023-03-04'
date_numeric = (pd.to_datetime(date_str) - dates.min()) / np.timedelta64(1, "D")
date_normalized = (date_numeric - data['DATE'].mean()) / data['DATE'].std()
input_tensor = torch.Tensor(date_normalized)

# Use the model to predict the output
with torch.no_grad():
    output_tensor = model(input_tensor)

# Convert the output to numpy array and denormalize
output_array = output_tensor.numpy()
output_array = output_array * data[['PRCP', 'TMAX', 'TMIN']].std().values + data[['PRCP', 'TMAX', 'TMIN']].mean().values

# Print the predicted output
print('PRCP prediction:', output_array[0])
print('TMAX prediction:', output_array[1])
print('TMIN prediction:', output_array[2])