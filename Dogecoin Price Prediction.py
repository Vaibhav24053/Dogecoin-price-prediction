#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


# In[17]:


data = pd.read_csv("DOGE-USD.csv")
data.head()


# In[18]:


data.corr()


# In[19]:


data['Date'] = pd.to_datetime(data['Date'],
							infer_datetime_format=True)
data.set_index('Date', inplace=True)

data.isnull().any()


# In[20]:


data.isnull().sum()


# In[21]:


data = data.dropna()


# In[22]:


data.describe()


# In[23]:


plt.figure(figsize=(20, 7))
x = data.groupby('Date')['Close'].mean()
x.plot(linewidth=2.5, color='b')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title("Date vs Close of 2021")


# In[24]:


data["gap"] = (data["High"] - data["Low"]) * data["Volume"]
data["y"] = data["High"] / data["Volume"]
data["z"] = data["Low"] / data["Volume"]
data["a"] = data["High"] / data["Low"]
data["b"] = (data["High"] / data["Low"]) * data["Volume"]
abs(data.corr()["Close"].sort_values(ascending=False))


# In[25]:


data = data[["Close", "Volume", "gap", "a", "b"]]
data.head()


# In[26]:


df2 = data.tail(30)
train = df2[:11]
test = df2[-19:]

print(train.shape, test.shape)


# In[27]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(endog=train["Close"], exog=train.drop(
	"Close", axis=1), order=(2, 1, 1))
results = model.fit()
print(results.summary())


# In[28]:


start = 11
end = 29
predictions = results.predict(
	start=start,
	end=end,
	exog=test.drop("Close", axis=1))
predictions


# In[29]:


test["Close"].plot(legend=True, figsize=(12, 6))
predictions.plot(label='TimeSeries', legend=True)


# In[ ]:




