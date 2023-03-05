import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("temp_by_months.csv")

# Convert the date column to a datetime object
data['date'] = pd.to_datetime(data['date'], format='%Y%m')

# Set the date column as the DataFrame's index
data.set_index('date', inplace=True)

# Define the ARIMA model with p=1, d=1, and q=1 for both temperature and precipitation
model_temp = ARIMA(data['temp'], order=(1, 1, 1))
model_precip = ARIMA(data['precip'], order=(1, 1, 1))

# Fit the models to the data
model_temp_fit = model_temp.fit()
model_precip_fit = model_precip.fit()

# Generate predictions for the next 12 months and store them in 2D arrays
n_steps = 12
temp_pred_2d = np.zeros((n_steps, 1))
precip_pred_2d = np.zeros((n_steps, 1))
for i in range(n_steps):
    temp_pred_2d[i, 0] = model_temp_fit.forecast()[0]
    precip_pred_2d[i, 0] = model_precip_fit.forecast()[0]

# Print the 2D arrays of predictions
print("Temperature predictions:")
print(temp_pred_2d)
print("Precipitation predictions:")
print(precip_pred_2d)