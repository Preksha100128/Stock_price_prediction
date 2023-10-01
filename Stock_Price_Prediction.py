#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[4]:


dataset = pd.read_excel("Stock Price Prediction.xlsx")


# In[16]:


dataset.head()


# In[17]:


dataset['Date'] = pd.to_datetime(dataset.Date)


# In[18]:


dataset.shape


# In[22]:


dataset.head(10)


# In[23]:


dataset.isnull().sum()


# In[24]:


dataset.isna().any()


# In[25]:


dataset.info()


# In[26]:


dataset.describe()


# In[27]:


print(len(dataset))


# In[28]:


dataset['Open'].plot(figsize=(16,6))


# In[29]:


x = dataset[['Open','High','Low','Volume']]
y = dataset['Close']


# In[30]:


from sklearn.model_selection import train_test_split
X_train , X_test,y_train ,y_test = train_test_split(x,y, random_state = 0)


# In[31]:


X_train.shape


# In[32]:


X_test.shape


# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,accuracy_score
regressor = LinearRegression()


# In[34]:


regressor.fit(X_train,y_train)


# In[35]:


print(regressor.coef_)


# In[36]:


predict=regressor.predict(X_test)


# In[37]:


print(X_test)


# In[38]:


predict.shape


# In[39]:


dframe=pd.DataFrame(y_test,predict)


# In[40]:


dfr=pd.DataFrame({'Actual Price':y_test,'predicted Price':predict})



# In[41]:


print(dfr)


# In[42]:


dfr.head(25)


# In[43]:


from sklearn.metrics import confusion_matrix, accuracy_score



# In[44]:


regressor.score(X_test,y_test)


# In[45]:


import math


# In[46]:


print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,predict))


# In[47]:


print('Mean Squared Error:',metrics.mean_squared_error(y_test,predict))


# In[48]:


print('Root Mean Squared Error:',math.sqrt(metrics.mean_squared_error(y_test,predict)))


# In[49]:


graph=dfr.head(20)


# In[50]:


graph.plot(kind='bar')


# In[ ]:




