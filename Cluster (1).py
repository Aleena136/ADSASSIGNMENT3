#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
df_inflation = pd.read_csv(r"C:\Users\VICTUS\Downloads\API_FP.CPI.TOTL.ZG_DS2_en_csv_v2_5447813.csv", skiprows=4)
df_gdp = pd.read_csv(r"C:\Users\VICTUS\Downloads\API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5358417.csv", skiprows=4)

# Extract the relevant data
df_inflation = df_inflation[['Country Name', '2016', '2017', '2018', '2019']]
df_gdp = df_gdp[['Country Name', '2016', '2017', '2018', '2019']]

# Merge the dataframes
df = pd.merge(df_inflation, df_gdp, on='Country Name', how='inner')
df = df.dropna()

# Set the index to be the country name
df.set_index('Country Name', inplace=True)

# Standardize the data
scaler = StandardScaler()
df_norm = scaler.fit_transform(df)

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_norm)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia,marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Cluster the data using K-means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df_norm)
labels = kmeans.predict(df_norm)

# Add the cluster labels as a new column to the dataframe
df['Cluster'] = labels

# Plot the clusters
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis')
plt.xlabel('Inflation, consumer prices (annual %))')
plt.ylabel('GDP per capita (current US$)')
plt.title('Clusters of Countries based on Inflation, consumer prices and GDP per capita')
plt.show()

# Plot the cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.xlabel('Inflation, consumer prices (annual %)')
plt.ylabel('GDP per capita (current US$)')
plt.title('Clusters of Countries based on Inflation, consumer prices and GDP per capita')
plt.show()

