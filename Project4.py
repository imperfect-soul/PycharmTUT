import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Define the paths to the dataset
train_path = "C:/Users/Admin/PycharmProjects/PycharmTut/titanic_datasets/train.csv"  # Update this path
test_path = "C:/Users/Admin/PycharmProjects/PycharmTut/titanic_datasets/test.csv"    # Update this path

# Check if the files exist
if not os.path.exists(train_path):
    print(f"Error: The file {train_path} does not exist.")
    exit(1)

if not os.path.exists(test_path):
    print(f"Error: The file {test_path} does not exist.")
    exit(1)

# Load the dataset
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Combine datasets for cleaning
df = pd.concat([train_data, test_data], ignore_index=True, sort=False)

# Display initial rows
print(df.head())

# Data Cleaning
# Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

# Feature Engineering
# Create FamilySize feature
df['FamilySize'] = df['SibSp'] + df['Parch']

# Extract Title from Name
df['Title'] = df['Name'].apply(lambda x: re.search(r' ([A-Za-z]+)\.', x).group(1) if re.search(r' ([A-Za-z]+)\.', x) else "")
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'noble')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Mrs')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Encode categorical variables
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Prepare Data for Modeling
# Define features and target variable
X = df.drop(columns=['Survived'])
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
# Initialize and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
dt_predictions = dt_model.predict(X_test)

# Evaluate the model
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")

# Train Linear Regression Model
# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test)

# Evaluate the model using Mean Squared Error
lr_mse = mean_squared_error(y_test, lr_predictions)
print(f"Linear Regression MSE: {lr_mse:.2f}")

# Hyperparameter Tuning (Optional)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters for Decision Tree: {grid_search.best_params_}")

# Evaluate the best model from GridSearchCV
best_dt_model = grid_search.best_estimator_
best_dt_predictions = best_dt_model.predict(X_test)
best_dt_accuracy = accuracy_score(y_test, best_dt_predictions)
print(f"Best Decision Tree Accuracy: {best_dt_accuracy:.2f}")