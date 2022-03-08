#!/usr/bin/env python
# coding: utf-8

# # NAME: TEJAS SURESH YADAV

# # ROLL NO.: 31283

# # BATCH: N2

# # ASSIGNMENT 04 

# # LINEAR REGRESSION

# In[ ]:





# In[2]:


import pandas as pd
import numpy as np


# # READING THE DATA

# In[3]:


df = pd.read_csv('housing.csv')
df


# In[4]:


df.describe()


# In[5]:


df.isnull().count()


# In[6]:


df.isnull()


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# In[8]:


pip install sklearn


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# In[11]:


# MEDV AND LSTAT ARE NEGATIVELY CORELATED.

x=df['MEDV']
y=df['LSTAT']
corelation = y.corr(x)
corelation


# In[12]:


plt.scatter(x,y)
plt.plot()


# In[ ]:





# # CORELATION MATRIX

# In[13]:


import seaborn as sns


# In[14]:


fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(df.corr(), annot=True)


# In[15]:


from sklearn.model_selection import train_test_split
x=df.loc[:, df.columns != 'MEDV']
y=df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)


# In[16]:


reg = LinearRegression()


# In[17]:


reg.fit(X_train, y_train)


# In[18]:


plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')


# In[19]:


reg.predict(X_test)


# In[20]:


reg.score(X_test, y_test)


# In[21]:


y_test


# In[ ]:





# # VISUALISING DATA

# In[22]:


# MEDV AND RM ARE NEGATIVELY CORELATED.

x=df['MEDV']
y=df['RM']
corelation = y.corr(x)
corelation


# In[23]:


plt.scatter(x,y)
plt.plot()


# In[24]:


import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['MEDV'])
plt.subplot(1,2,2)
sns.distplot(df['LSTAT'])
plt.show()


# In[25]:


warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['MEDV'])
plt.subplot(1,2,2)
sns.distplot(df['RM'])
plt.show()


# In[ ]:





# # REMOVING OUTLIERS OF RM AND LSTAT

# In[26]:


print("Highest allowed",df['RM'].mean() + 3*df['RM'].std())
print("Lowest allowed",df['RM'].mean() - 3*df['RM'].std())


# In[27]:


print("Highest allowed",df['LSTAT'].mean() + 3*df['LSTAT'].std())
print("Lowest allowed",df['LSTAT'].mean() - 3*df['LSTAT'].std())


# In[28]:


upper_limit_of_LSTAT = df['LSTAT'].mean() + 3*df['LSTAT'].std()
lower_limit_of_LSTAT = df['LSTAT'].mean() - 3*df['LSTAT'].std()

upper_limit_of_RM = df['RM'].mean() + 3*df['RM'].std()
lower_limit_of_RM = df['RM'].mean() - 3*df['RM'].std()


# In[29]:


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


# In[30]:


new_df1 = new_df[(new_df['RM'] < upper_limit_of_RM) & (new_df['RM'] > lower_limit_of_RM)]
new_df1


# In[31]:


new_df1['LSTAT'].describe()


# In[32]:


LSTAT_1 = new_df1['LSTAT']

x=df['MEDV'].iloc[1:495]
y=LSTAT_1
plt.scatter(x,y)
plt.show()


# In[33]:


new_df1['RM'].describe()


# In[34]:


RM_1 = new_df1['RM']

x=df['MEDV'].iloc[1:495]
y=RM_1
plt.scatter(x,y)
plt.show()


# In[35]:


x=new_df1[['RM', 'LSTAT']]
y=new_df1['MEDV']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)


# In[36]:


reg = LinearRegression()


# In[37]:


reg.fit(X_train, y_train)


# In[38]:


reg.predict(X_test)


# In[39]:


y_test


# In[40]:


reg.score(X_test, y_test)


# In[41]:


x=new_df1.loc[:, df.columns != 'MEDV']
y=new_df1['MEDV']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[42]:


reg.fit(X_train, y_train)


# In[43]:


y_pred=reg.predict(X_test)


# In[44]:


y_test


# In[45]:


reg.score(X_test, y_test)


# In[46]:


import math
y_actual = y_test
y_predicted = y_pred
 
MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:\n")
print(RMSE)


# In[47]:


import sklearn as sk

mse = sk.metrics.mean_squared_error(y_actual, y_predicted)
print("MSE: ", mse)

rmse = math.sqrt(mse)


print("RMSE",rmse)


# In[48]:


mae=sk.metrics.mean_absolute_error(y_actual, y_predicted)
print("MAE: ",mae)


# In[49]:


predictions = reg.predict(X_test)
d = pd.DataFrame({'ytest':y_test, 'predictions':y_pred})
d


# In[ ]:





# # REMOVING OUTLIERS OF MEDV

# In[51]:


print("Highest allowed MEDV: ",df['MEDV'].mean() + 3*df['MEDV'].std())
print("Lowest allowed MEDV: ",df['MEDV'].mean() - 3*df['MEDV'].std())


# In[53]:


upper_limit_of_MEDV = df['MEDV'].mean() + 3*df['MEDV'].std()
lower_limit_of_MEDV = df['MEDV'].mean() - 3*df['MEDV'].std()


# In[54]:


new_df2 = new_df1[(new_df1['MEDV'] < upper_limit_of_MEDV) & (new_df1['MEDV'] > lower_limit_of_MEDV)]
new_df2


# In[55]:


new_df2.describe()


# In[ ]:





# # PREDICTING THE VALUES

# In[109]:


# FOR RATIO 75:25

x1=new_df2.loc[:, new_df2.columns != 'MEDV']
x=x1.loc[:, x1.columns != 'CHAS']
y=new_df2['MEDV']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[110]:


reg.fit(X_train, y_train)


# In[111]:


y_pred=reg.predict(X_test)


# In[112]:


reg.score(X_test, y_test)


# In[113]:


predictions = reg.predict(X_test)
d = pd.DataFrame({'ytest':y_test, 'predictions':y_pred})
d


# # CALCULATING THE ERRORS

# In[114]:


print("R2 Score: ", sk.metrics.r2_score(y_actual, y_predicted))


# In[120]:


mse = sk.metrics.mean_squared_error(y_actual, y_predicted)
print("MSE: ", mse)

rmse = math.sqrt(mse)


print("RMSE",rmse)
mae=sk.metrics.mean_absolute_error(y_actual, y_predicted)
print("MAE: ",mae)


# In[ ]:





# # PLOTTING THE   "Y = MX + C"    LINE

# In[119]:


sns.lmplot(data=d, x="ytest", y="predictions")


# In[132]:


X = d['ytest']
Y = d['predictions']
plt.scatter(X, Y)


# In[133]:


model = np.polyfit(X, Y, 1)


# In[134]:


model


# In[135]:


# model[0] = m (slope)  model[1] = c (intercept)

print("Slope: ", model[0])
print("Intercept: ", model[1])


# In[136]:


print("The equation of line in the form of y = mx+c  is: y = ", model[0], "x + ", model[1])


# In[ ]:





# # FOR SPLIT 70:30

# In[122]:


x1=new_df2.loc[:, new_df2.columns != 'MEDV']
x=x1.loc[:, x1.columns != 'CHAS']
y=new_df2['MEDV']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

reg.fit(X_train, y_train)


# In[123]:


y_pred=reg.predict(X_test)


# In[124]:


reg.score(X_test, y_test)


# In[125]:


predictions = reg.predict(X_test)
d = pd.DataFrame({'ytest':y_test, 'predictions':y_pred})
d


# In[126]:


print("R2 Score: ", sk.metrics.r2_score(y_actual, y_predicted))
mse = sk.metrics.mean_squared_error(y_actual, y_predicted)
print("MSE: ", mse)
rmse = math.sqrt(mse)
print("RMSE",rmse)
mae=sk.metrics.mean_absolute_error(y_actual, y_predicted)
print("MAE: ",mae)


# In[ ]:





# # FOR SPLIT 80:20

# In[127]:


x1=new_df2.loc[:, new_df2.columns != 'MEDV']
x=x1.loc[:, x1.columns != 'CHAS']
y=new_df2['MEDV']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg.fit(X_train, y_train)


# In[128]:


y_pred=reg.predict(X_test)
reg.score(X_test, y_test)


# In[129]:


predictions = reg.predict(X_test)
d = pd.DataFrame({'ytest':y_test, 'predictions':y_pred})
d


# In[130]:


print("R2 Score: ", sk.metrics.r2_score(y_actual, y_predicted))
mse = sk.metrics.mean_squared_error(y_actual, y_predicted)
print("MSE: ", mse)
rmse = math.sqrt(mse)
print("RMSE",rmse)
mae=sk.metrics.mean_absolute_error(y_actual, y_predicted)
print("MAE: ",mae)


# In[ ]:





# In[ ]:





# In[138]:


x1=new_df2[['LSTAT', 'RM']]
y=new_df2['MEDV']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

reg.fit(X_train, y_train)


# In[139]:


y_pred=reg.predict(X_test)
reg.score(X_test, y_test)


# In[140]:


predictions = reg.predict(X_test)
d = pd.DataFrame({'ytest':y_test, 'predictions':y_pred})
d


# In[141]:


print("R2 Score: ", sk.metrics.r2_score(y_actual, y_predicted))
mse = sk.metrics.mean_squared_error(y_actual, y_predicted)
print("MSE: ", mse)
rmse = math.sqrt(mse)
print("RMSE",rmse)
mae=sk.metrics.mean_absolute_error(y_actual, y_predicted)
print("MAE: ",mae)


# In[ ]:



