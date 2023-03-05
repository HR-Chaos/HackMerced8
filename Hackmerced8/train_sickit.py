import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pickle


'''
removed: add later
'''

# Load the data from a CSV file
data = pd.read_csv('train_data1.csv')

# Convert categorical variables to numerical values
data = pd.get_dummies(data)

# Remove missing values
data.dropna(inplace=True)

# Split the data into training and testing sets
X = data.drop('yield', axis=1)
X = data.drop('year', axis=1)
y = data['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the decision tree regressor object
dt_reg = DecisionTreeRegressor(random_state=42)
# # Train a random forest model
# rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the decision tree on the training data
dt_reg.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = dt_reg.predict(X_test)

# Evaluate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
print(y_test, ' : ',y_pred)

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(dt_reg, filled=True)
plt.savefig('decision_tree1.png')
plt.show()


with open('decision_tree.pkl', 'wb') as f:
    pickle.dump(dt_reg, f)

# Load the saved model from a file
# with open('decision_tree.pkl', 'rb') as f:
#     dt_reg = pickle.load(f)