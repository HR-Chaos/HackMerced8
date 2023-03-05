import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pickle


'''
This is the test file for the random forest model
'''

print('\n\n\n-------------------------------------------------------------------\n\n\n')

print('Loading dataset...')
# Load the data from a CSV file
data = pd.read_csv('train_data1.csv')

# Convert categorical variables to numerical values
data = pd.get_dummies(data)

# Remove missing values
data.dropna(inplace=True)

# Split the data into training and testing sets
print('Splitting data into training and testing sets...')
X = data.drop(['yield'], axis=1)
y = data['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved model from a file
print('Loading saved model from file...')
with open('Random_Forest.pkl', 'rb') as f:
    rf_reg = pickle.load(f)

# Evaluate the mean squared error
print('Evaluating the mean squared error...')
y_pred = rf_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Print the first 10 predictions
print('Printing the first 10 predictions...')
print('\t NOTE: The first 10 predictions are not from the training set.\n\n')

test_arr = y_test.to_numpy()
pred_arr = y_pred
print('Ground Truth \t Prediction')
for i in range(10):
    print(test_arr[i], '\t\t', pred_arr[i])

print('Printing the predicted yield from the predicted weather the next year...')
# unpack the weather data
weather = pd.read_csv('weather_predictions.csv')
# turn it into a 1D array
weather = weather.to_numpy().reshape(1, -1)
print(weather[0])
# predict the yield for the next year
next_year = rf_reg.predict(weather)
print('Predicted yield for next year: ', next_year)

# Visualize the decision tree
# plt.figure(figsize=(20,10))
# plot_tree(rf_reg, filled=True)
# plt.savefig('decision_tree2.png')
# plt.show()


print('\n\n\n-------------------------------------------------------------------\n\n\n')