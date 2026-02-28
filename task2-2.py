import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Titanic Dataset
data = pd.read_csv('titanic.csv')
print(data.head())
print(data.info())

# Step 2: Data Cleaning

# 2.1 Handle Missing Values
print(data.isnull().sum())
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop('Cabin', axis=1, inplace=True)
print(data.isnull().sum())

# 2.2 Remove Duplicates
data.drop_duplicates(inplace=True)

# 2.3 Correct Inconsistent Data Types or Formats
data['Survived'] = data['Survived'].astype('category')
data['Pclass'] = data['Pclass'].astype('category')

# Step 3: Feature Engineering

# 3.1 Create New Features
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['AgeCategory'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, 80], labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])
print(data[['Name', 'Title', 'Age', 'AgeCategory']].head())

# 3.2 Convert Categorical Features into Numerical Representations
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
data = pd.get_dummies(data, columns=['Title'], drop_first=True)
print(data.head())

# 3.3 Feature Scaling or Normalization
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])
print(data.head())