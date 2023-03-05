import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("temp_by_months.csv")

# Convert the date column to a datetime object
data['date'] = pd.to_datetime(data['date'], format='%Y%m')

# Set the date column as the DataFrame's index
data.set_index('date', inplace=True)

# Define the AR model with p=12 for both temperature and precipitation
p = 12
model_temp = AutoReg(data['temp'], lags=p)
model_precip = AutoReg(data['precip'], lags=p)

# Fit the models to the data
model_temp_fit = model_temp.fit()
model_precip_fit = model_precip.fit()

# Generate predictions for the next 12 months and store them in 2D arrays
n_steps = 12
temp_pred_2d = np.zeros((n_steps, 1))
precip_pred_2d = np.zeros((n_steps, 1))
for i in range(n_steps):
    temp_pred_2d[i, 0] = model_temp_fit.predict(start=len(data)+i, end=len(data)+i)[0]
    precip_pred_2d[i, 0] = model_precip_fit.predict(start=len(data)+i, end=len(data)+i)[0]

# Print the 2D arrays of predictions
print('\n\n\n-------------------------------------------------------------------\n\n\n')
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
print("Temperature predictions for next year (F):")
for i in range(12):
    print(months[i], ' : ', round(temp_pred_2d[i, 0],2))
print("\n\nPrecipitation predictions for next year (in):")
for i in range(12):
    if precip_pred_2d[i, 0] < 0:
        precip_pred_2d[i, 0] = 0
    print(months[i], ' : ', round(precip_pred_2d[i, 0],2))

# put it into a dimensional vector
vect = []
for i in range (24):
    if i < 12:
        vect.append(round(temp_pred_2d[i, 0],2))
    else:
        if precip_pred_2d[i-12, 0] < 0:
            precip_pred_2d[i-12, 0] = 0
        vect.append(round(precip_pred_2d[i-12, 0],2))

# print(vect)
np.savetxt("weather_predictions.csv", vect, delimiter=",")

print('\n\n\n-------------------------------------------------------------------\n\n\n')