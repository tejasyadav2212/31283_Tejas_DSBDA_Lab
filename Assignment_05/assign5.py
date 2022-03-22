#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Social_Network_Ads.csv")
df


# In[3]:


df.describe()


# In[4]:


new_df = df.drop(['User ID'], axis=1)
new_df


# In[5]:


# 0 - female;  1 - Male ....
new_df['Gender'].replace(['Female','Male'],[0,1], inplace = True)
new_df['Gender']


# In[6]:


new_df


# In[95]:


new_df['Purchased'].value_counts()


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# In[8]:


x = new_df['Age']
y = new_df['Purchased']
plt.scatter(x, y)
plt.show()


# In[9]:


x = new_df['EstimatedSalary']
y = new_df['Purchased']
plt.scatter(x, y)
plt.show()


# In[10]:


x = new_df['Gender']
y = new_df['Purchased']
plt.scatter(x, y)
plt.show()


# In[79]:


x=new_df.loc[:, new_df.columns != 'Purchased']
y=new_df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[80]:


X_train


# In[81]:


y_train


# In[82]:


X_test


# In[83]:


y_test


# In[84]:


logReg = LogisticRegression(C = 50, multi_class = 'multinomial',solver = 'saga', tol = 0.1)


# In[85]:


logReg.fit(X_train, y_train)


# In[86]:


logReg.fit(X_train, y_train)


# In[87]:


yp = logReg.predict(X_test)


# In[88]:


yp


# In[76]:


from sklearn.metrics import confusion_matrix


# In[78]:


cm = confusion_matrix(y_test, yp)
cm


# In[41]:


new_df.describe()


# In[42]:


y_test


# In[43]:


logReg.score(X_test, y_test)


# In[44]:


new_df2 = new_df[['Age', 'EstimatedSalary', 'Purchased']]
new_df2


# In[59]:


x=new_df['Gender']
x = x.reshape(-1, 1)
y=new_df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[57]:


logReg = LogisticRegression()


# In[58]:


logReg.fit(X_train, y_train)


# In[48]:


logReg.fit(X_train, y_train)


# In[51]:


yp = logReg.predict(X_test)


# In[52]:


yp


# In[96]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(new_df))


# In[97]:


print(scaler.data_max_)


# In[98]:


print(scaler.transform(new_df))


# In[104]:


new_df


# In[ ]:



