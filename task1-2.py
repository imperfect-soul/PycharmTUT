import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('dietary_weight_control_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check the dataset's information
print(data.info())

# Check for any missing values
print(data.isnull().sum())

# Visualize the relationship between Sales and Advertising
sns.scatterplot(x='Advertising', y='Sales', data=data)
plt.title('Sales vs Advertising')
plt.xlabel('Advertising')
plt.ylabel('Sales')
plt.show()

# Define the features and target variable
X = data[['Advertising']]  # Features
y = data['Sales']          # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the regression line
plt.scatter(X_test, y_test, color='blue', label='Actual Sales')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Sales')
plt.title('Sales vs Advertising (Test Set)')
plt.xlabel('Advertising')
plt.ylabel('Sales')
plt.legend()
plt.show()