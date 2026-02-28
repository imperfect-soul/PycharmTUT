import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a synthetic dataset
# For demonstration, let's create a dataset with advertising spend and sales
data = {
    'Advertising': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'Sales': [10, 20, 25, 30, 35, 40, 45, 50, 55, 60]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the dataset
print("Dataset:")
print(df)

# Step 2: Visualize the data
plt.scatter(df['Advertising'], df['Sales'], color='blue')
plt.title('Sales vs Advertising')
plt.xlabel('Advertising Spend (in thousands)')
plt.ylabel('Sales (in thousands)')
plt.grid()
plt.show()

# Step 3: Prepare the data for training
X = df[['Advertising']]  # Feature
y = df['Sales']          # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Step 7: Visualize the regression line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Sales vs Advertising with Regression Line')
plt.xlabel('Advertising Spend (in thousands)')
plt.ylabel('Sales (in thousands)')
plt.legend()
plt.grid()
plt.show()