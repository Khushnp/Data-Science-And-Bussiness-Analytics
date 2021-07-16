#!/usr/bin/env python
# coding: utf-8

# # NAME: Khush N. Pachghare
# 
# ##  The Spark Foundation internship program

# ### TASK: Predict the percentage of an student based on the no. of study hours. 
# ### dataset: http://bit.ly/w-data

# In[1]:


#importing the required lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading the dataset
data = pd.read_csv("Supervised_ML_Data.csv")


# In[3]:


#checking the dataset by printing its first 10 rows
data.head(10)


# ## Ploting graphs for Analysis of the Dataset

# In[4]:


data.plot.scatter(x='Hours',y='Scores')


# In[5]:


Hours = data.Hours
Scores = data.Scores

plt.plot(Hours, Scores,'g--')


# In[6]:


data.plot(x='Hours',y='Scores',style='1')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Scored')
plt.show()


# In[7]:


data.plot.bar(x='Hours',y='Scores')


# In[8]:


# data.sort_values(['Hours'],axis=0,
#                 ascending=[True],inplace=True)
# data.head()
# data.plot.bar(x='Hours',y='Scores')


# ## After Plotting the graphs as we observed that the dataset is correct and it is fine to go forward and apply regression algorithms and predict ans based on the dataset

# #### First splitting data into test and train

# In[9]:


X=data.iloc[:, :-1].values
y=data.iloc[:, -1].values
#print(x)


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# #### Now Training the data using algorithm

# In[11]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


#from sklearn.essemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42)

regressor.fit(X_train,y_train)
print("Training complete")


# #### Now our model is ready its time to test it

# In[12]:


print(y_test)
print("Prediction of scores")
y_pred = regressor.predict(X_test)
print(y_pred)


# In[13]:


df = pd.DataFrame({'Actual': y_test,'Prediction': y_pred})

df


# In[14]:


# Now its time to predict by custom input
hours=[[9.25]]
pred = regressor.predict(hours)
print(pred)


# ##### Evaluating the model (finding the absolute error)

# In[15]:


# this evaluation is without sorting the dataset
from sklearn import metrics
print('Mean Absolute error:',
      metrics.mean_absolute_error(y_test,y_pred))


# ## Our Model Accuracy is 4.183859899002975 by using Linear Regression

# In[ ]:




