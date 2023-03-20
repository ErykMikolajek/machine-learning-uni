#!/usr/bin/env python
# coding: utf-8

# In[57]:


import os
path = '/Users/eryk/Documents/Uczenie Maszynowe/lab01/data'

if not os.path.exists(path):
    os.mkdir(path)


# In[58]:


import urllib.request
urllib.request.urlretrieve('https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz', 'data/downloaded_files.tgz')


# In[59]:


import tarfile
tar = tarfile.open('data/downloaded_files.tgz', 'r:gz')
tar.extractall('./data')
tar.close()


# In[60]:


import gzip
with open('data/housing.csv', 'rb') as f_in, gzip.open('data/housing.csv.gz', 'wb') as f_out:
    f_out.writelines(f_in)


# In[61]:


os.remove('data/downloaded_files.tgz')


# In[62]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/housing.csv.gz')
df


# In[63]:


df.head()


# In[64]:


df.info()


# In[65]:


df.ocean_proximity.value_counts()


# In[66]:


df.ocean_proximity.describe()


# In[67]:


df.hist(bins=50, figsize=(20,15))
plt.savefig('data/obraz1.png')

# In[68]:


df.plot(kind="scatter", x="longitude", y="latitude",
alpha=0.1, figsize=(7,4))
plt.savefig('data/obraz2.png')

# In[69]:


df.plot(kind="scatter", x="longitude", y="latitude",
alpha=0.4, figsize=(7,3), colorbar=True,
s=df["population"]/100, label="population",
c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('data/obraz3.png')


# In[70]:


df.corr()["median_house_value"].sort_values(ascending=False)


# In[71]:


df.corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(columns={'index': 'atrybut', 'median_house_value': 'wspolczynnik_korelacji'}).to_csv(index=False, path_or_buf='data/korelacja.csv')


# In[72]:

import seaborn as sns
sns.pairplot(df)


# In[73]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set),len(test_set)


# In[74]:


train_set.head()


# In[75]:


test_set.head()


# In[76]:


train_set.corr()


# In[77]:


test_set.corr() # macierze korelacji dla train_set i test_set są bardzo podobne, to skutek odpowiednio losowego podzielenia tych zbiorów przez train_test_split


# In[ ]:

import pickle
with open('data/train_set.pkl', 'wb') as train_pickle, open('data/test_set.pkl', 'wb') as test_pickle:
    pickle.dump(train_set, train_pickle)
    pickle.dump(test_set, test_pickle)
    
    
# train_set.to_pickle('train_set.pkl')
# test_set.to_pickle('test_set.pkl')

