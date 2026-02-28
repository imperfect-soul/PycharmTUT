# Step 1: Setup and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# Load dataset
df = pd.read_csv('train.csv')
# Display first few rows
print(df.head())
# Display information about the dataframe
print(df.info())
# Display descriptive statistics
print(df.describe(include='all'))

# Step 2: Initial Exploration & Identifying Issues
# Calculate missing value percentages
missing_values = df.isnull().sum()
missing_percentages = (missing_values / len(df)) * 100
missing_data = pd.concat([missing_values, missing_percentages], axis=1)
missing_data.columns = ['Missing Count', 'Missing Percentage']
print(missing_data[missing_data['Missing Count'] > 0])
# Analyze data types
print(df.dtypes)
# Examine categorical features
for col in ['Sex', 'Embarked', 'Pclass']:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())
    print(f"{col} unique values:")
    print(df[col].unique())
# Basic statistics for numerical features
for col in ['Age', 'Fare', 'SibSp', 'Parch']:
    print(f"\n{col} statistics:")
    print(df[col].describe())
    # Create box plot to identify potential outliers
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col])
    plt.title(f"{col} Distribution")
    plt.show()

    # Step 3: Handling Missing Data

    # Age: Impute missing values using the median age grouped by Pclass and Sex
    age_median = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
    df['Age'] = df['Age'].fillna(age_median)

    # Check if Age still has missing values
    print(f"Missing values in Age after imputation: {df['Age'].isnull().sum()}")

    # If still missing, fill with overall median
    if df['Age'].isnull().sum() > 0:
        df['Age'] = df['Age'].fillna(df['Age'].median())

    # Cabin: Extract deck information or create 'Unknown'
    df['Deck'] = df['Cabin'].str[0]  # Get first letter which represents the deck
    df['Deck'] = df['Deck'].fillna('U')  # U for Unknown

    # Embarked: Fill with the most frequent value (mode)
    most_common_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(most_common_embarked)

    # Check if there are any remaining missing values in the columns we filled
    print(df[['Age', 'Embarked', 'Deck']].isnull().sum())

    # Step 4: Feature Engineering
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 for the passenger themselves
    # Create a binary feature for if the passenger is traveling alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    # Extract titles from names
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.')
    # Group rare titles
    rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
    df.loc[df['Title'].isin(rare_titles), 'Title'] = 'Rare'
    # Map common titles
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    # Create age bins
    df['AgeBin'] = pd.cut(df['Age'],
                          bins=[0, 12, 20, 40, 60, 100],
                          labels=['Child', 'Teenager', 'Adult', 'Middle Aged', 'Senior'])
    # Create fare bins
    df['FareBin'] = pd.qcut(df['Fare'],
                            q=4,
                            labels=['Low', 'Medium', 'High', 'Very High'])
    # Display new features
    print(df[['FamilySize', 'IsAlone', 'Title', 'AgeBin', 'FareBin', 'Deck']].head())


    # Step 5: Feature Transformation & Encoding
    # Convert Sex to binary (0 for female, 1 for male)
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    # One-hot encode categorical features
    categorical_features = ['Embarked', 'Title', 'Deck', 'AgeBin', 'FareBin']
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Ensure all features are numeric
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            try:
                df_encoded[col] = pd.to_numeric(df_encoded[col])
            except:
                print(f"Could not convert {col} to numeric. Converting to categorical.")
                df_encoded[col] = pd.factorize(df_encoded[col])[0]

    print(df_encoded.dtypes)

    # Step 6: Feature Selection / Dropping Redundant Columns
    # List of columns to drop
    columns_to_drop = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']
    # Drop columns
    df_final = df_encoded.drop(columns=columns_to_drop)
    # Display the final list of features
    print(f"Final features ({len(df_final.columns)} total):")
    print(df_final.columns.tolist())
    print(df_final.head())


    # Step 7: Final Review
    # Check for any remaining missing values
    print("Missing values in final dataset:")
    print(df_final.isnull().sum().sum())
    # Check data types
    print("\nData types in final dataset:")
    print(df_final.dtypes)
    # Basic statistics of final dataset
    print("\nStatistics of final dataset:")
    print(df_final.describe())
    # Define X (features) and y (target)
    X = df_final.drop('Survived', axis=1)
    y = df_final['Survived']
    print(f"\nTraining data shape: {X.shape}")
    print(f"Target variable distribution:\n{y.value_counts(normalize=True)}")
    print("Dataset is now clean, engineered, and ready for machine learning models.")