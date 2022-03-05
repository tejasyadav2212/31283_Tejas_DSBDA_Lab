#code of 4 th assignment


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('housing.csv')
df


# In[3]:


df.describe()


# In[4]:


df.isnull().count()


# In[5]:


df.isnull()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# In[8]:


pip install sklearn


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# In[8]:


# MEDV AND LSTAT ARE NEGATIVELY CORELATED.

x=df['MEDV']
y=df['LSTAT']
corelation = y.corr(x)
corelation


# In[9]:


plt.scatter(x,y)
plt.plot()


# In[10]:


import seaborn as sns


# In[11]:


fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(df.corr(), annot=True)


# In[21]:


from sklearn.model_selection import train_test_split
x=df.loc[:, df.columns != 'MEDV']
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)


# In[22]:


reg = LinearRegression()


# In[23]:


reg.fit(X_train, y_train)


# In[24]:


plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')


# In[25]:


reg.predict(X_test)


# In[26]:


reg.score(X_test, y_test)


# In[27]:


y_test


# In[19]:


# MEDV AND RM ARE NEGATIVELY CORELATED.

x=df['MEDV']
y=df['RM']
corelation = y.corr(x)
corelation


# In[20]:


plt.scatter(x,y)
plt.plot()


# In[28]:


import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['MEDV'])
plt.subplot(1,2,2)
sns.distplot(df['LSTAT'])
plt.show()


# In[29]:


warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['MEDV'])
plt.subplot(1,2,2)
sns.distplot(df['RM'])
plt.show()


# In[30]:


print("Highest allowed",df['RM'].mean() + 3*df['RM'].std())
print("Lowest allowed",df['RM'].mean() - 3*df['RM'].std())


# In[31]:


print("Highest allowed",df['LSTAT'].mean() + 3*df['LSTAT'].std())
print("Lowest allowed",df['LSTAT'].mean() - 3*df['LSTAT'].std())


# In[32]:


upper_limit_of_LSTAT = df['LSTAT'].mean() + 3*df['LSTAT'].std()
lower_limit_of_LSTAT = df['LSTAT'].mean() - 3*df['LSTAT'].std()

upper_limit_of_RM = df['RM'].mean() + 3*df['RM'].std()
lower_limit_of_RM = df['RM'].mean() - 3*df['RM'].std()


# In[39]:


# LSTAT_1 = []
# LSTAT_1 = np.where(
#     df['LSTAT']>upper_limit_of_LSTAT,
#     upper_limit_of_LSTAT,
#     np.where(
#         df['LSTAT']<lower_limit_of_LSTAT,
#         lower_limit_of_LSTAT,
#         df['LSTAT']
#     )
# )

new_df = df[(df['LSTAT'] < upper_limit_of_LSTAT) & (df['LSTAT'] > lower_limit_of_LSTAT)]
new_df


# In[40]:


new_df1 = new_df[(new_df['RM'] < upper_limit_of_RM) & (new_df['RM'] > lower_limit_of_RM)]
new_df1


# In[41]:


new_df1['LSTAT'].describe()


# In[45]:


LSTAT_1 = new_df1['LSTAT']

x=df['MEDV'].iloc[1:495]
y=LSTAT_1
plt.scatter(x,y)
plt.show()


# In[46]:


new_df1['RM'].describe()


# In[47]:


RM_1 = new_df1['RM']

x=df['MEDV'].iloc[1:495]
y=RM_1
plt.scatter(x,y)
plt.show()


# In[57]:


x=new_df1[['RM', 'LSTAT']]
y=new_df1['MEDV']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)


# In[58]:


reg = LinearRegression()


# In[59]:


reg.fit(X_train, y_train)


# In[60]:


reg.predict(X_test)


# In[61]:


y_test


# In[62]:


reg.score(X_test, y_test)


# In[113]:


x=new_df1.loc[:, df.columns != 'MEDV']
y=new_df1['MEDV']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[114]:


reg.fit(X_train, y_train)


# In[115]:


reg.predict(X_test)


# In[116]:


y_test


# In[117]:


reg.score(X_test, y_test)


# In[79]:


reg.score()


# In[ ]:



