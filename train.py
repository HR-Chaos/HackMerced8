import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from model import Model

data = pd.read_csv('Weather_data_Fresno_cleaned.csv')

# Convert date to numerical format
data['DATE'] = pd.to_datetime(data['DATE'])
data['DATE'] = data['DATE'].dt.strftime('%Y%m%d').astype(int)

# Normalize the data
data = (data - data.mean()) / data.std()

# Split the data into input and output
x = data['DATE'].values.reshape(-1, 1)
y = data[['PRCP', 'TMAX', 'TMIN']].values

model = Model()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    # Convert data to tensors
    inputs = torch.Tensor(x)
    targets = torch.Tensor(y)

    # Forward pass
    outputs = model(inputs)

    # Compute loss
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))
        
# save the model
torch.save(model.state_dict(), 'model.pt')

# date = '2023-03-04'
# date = pd.to_datetime(date).strftime('%Y%m%d')

# # Normalize date
# date = (date - data['DATE'].mean()) / data['DATE'].std()

# # Convert to tensor and predict
# date = torch.Tensor(date)
# prediction = model(date)

# # Denormalize prediction
# prediction = prediction.detach().numpy()
# prediction = prediction * data[['PRCP', 'TMAX', 'TMIN']].std().values + data[['PRCP', 'TMAX', 'TMIN']].mean().values

# print('PRCP prediction:', prediction[0])
# print('TMAX prediction:', prediction[1])
# print('TMIN prediction:', prediction[2])