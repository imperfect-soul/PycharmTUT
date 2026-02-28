import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Titanic Dataset
data = pd.read_csv('titanic.csv')

# Step 2: Data Cleaning

# Handle Missing Values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop('Cabin', axis=1, inplace=True)

# Remove Duplicates
data.drop_duplicates(inplace=True)

# Correct Inconsistent Data Types or Formats
data['Survived'] = data['Survived'].astype('category')
data['Pclass'] = data['Pclass'].astype('category')

# Step 3: Feature Engineering

# Create New Features
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['AgeCategory'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, 80], labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])

# Convert Categorical Features into Numerical Representations
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
data = pd.get_dummies(data, columns=['Title'], drop_first=True)

# Feature Scaling
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Step 4: Data Preprocessing

# Define features and target variable
X = data.drop(columns=['Survived', 'Name', 'Ticket', 'PassengerId'])  # Drop non-feature columns
y = data['Survived']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training

# Train a Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Train a Logistic Regression model (as a substitute for Linear Regression)
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)

# Step 6: Model Evaluation

# Decision Tree Predictions
y_pred_tree = decision_tree_model.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

# Logistic Regression Predictions
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

# Print Evaluation Metrics
print(f'Decision Tree Accuracy: {accuracy_tree:.4f}')
print(f'Logistic Regression Accuracy: {accuracy_logistic:.4f}')