#!/usr/bin/env python
# coding: utf-8

# # NAME: Khush N. Pachghare
# 
# ##  The Spark Foundation internship program

# ### TASK 2: From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.
# ### Dataset : https://bit.ly/3kXTdox

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans


# In[2]:


#reading the dataset
iris = pd.read_csv("iris.csv")


# In[12]:


#Viewing the dataset
iris.head()


# In[5]:


#checking for the null values
iris.isnull().sum()


# In[6]:


#extracting the information about the dataset
iris.info()


# In[9]:


#describing the dataset
iris.describe()


# In[11]:


#checking for shape of the dataset
iris.shape


# ### Data Preprocessing

# In[13]:


#droping unwanted columns
x=iris.drop(['Id','Species'],axis=1)
x.head()


# ### K Means Clustering
# #### A Cluster is a collection of data points aggregated together due to similarities in between.
# #### K Means algorithm identifis k number of centroids, and then allocate every data point to the nearest clusters , while keeping the centroids as small as possible.
# ### Let us find out number of clusters for k-means classification.

# In[14]:


x= iris.iloc[:,[0,1,2,3]].values 


# ### Elbow Method runs K-Means clustering on dataset on number of values for k and then for each values of k computes an average scores for all clusters

# In[15]:


wcss=[]

for i in range(1,10):
    kmeans=KMeans(i)
    kmeans.fit(x)
    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)
    
wcss


# In[16]:


#Plotting a result on to a line graph, allowing us to observe 'The elbow '
nu_clusters=range(1,10)
plt.plot(nu_clusters,wcss,'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Clusters Sum of squares')


# In[17]:


#From the upper graph we came to know that elbow is at 3 so there will be three clusters.
kmeans = KMeans(n_clusters=3,init= 'k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(x)


# In[18]:


#Clusters
y_kmeans


# ### Visualisation of Clusters

# In[19]:


plt.figure(figsize=(10,6))
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], s=100,c='blue',label='Iris-versicolar')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], s=100,c='green',label='Iris-verginica')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100,c='yellow',label='Centroids')
plt.legend()


# ### And by this way we are able to see the three clusters from the given dataset and visualised it.

# In[ ]:




