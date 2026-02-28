import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the Titanic dataset
titanic_dataset = pd.read_csv("titanic.csv")

# Preprocessing
titanic_dataset.fillna(0, inplace=True)
titanic_dataset['Sex'] = titanic_dataset['Sex'].map({'male': 0, 'female': 1})

# Select relevant features
features = titanic_dataset[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values

# Normalize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features_scaled)

# Implement K-means clustering
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(reduced_features)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering on Titanic Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()