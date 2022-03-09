#!/usr/bin/env python
# coding: utf-8

# # Regression

# In[19]:


import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
import numpy as np
from sklearn.metrics import mean_squared_error


# In[2]:


df_dataset = pd.read_excel("petrol_consumption.xlsx")


# In[3]:


df_dataset


# In[4]:


y = df_dataset["Petrol_Consumption"]
X = df_dataset.drop(["Petrol_Consumption"], axis = 1)


# In[5]:


X


# In[6]:


y


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=43)


# In[9]:


len(X_train)


# In[10]:


len(y_train)


# In[11]:


len(X_test)


# In[12]:


len(y_test)


# In[14]:


rf_regressor = RandomForestRegressor().fit(X_train, y_train)


# In[15]:


y_pred = rf_regressor.predict(X_test)


# In[21]:


y_pred


# In[22]:


y_test


# In[20]:


rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))

print("\nRMSE: ", rmse)

