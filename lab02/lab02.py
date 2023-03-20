#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml 
mnist = fetch_openml('mnist_784', version=1)


# In[2]:


import numpy as np
print((np.array(mnist.data.loc[0]).reshape(28, 28) > 0).astype(int))


# In[3]:


X = mnist.data
y = mnist.target


# In[4]:


y = y.sort_values()


# In[5]:


X = X.reindex(y.index)


# In[6]:


X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[7]:


print(y_train.unique())
print(y_test.unique())


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=42)


# In[9]:


print(y_train.unique().sort_values())
print(y_test.unique().sort_values())


# In[10]:


y0_train = (y_train == '0')
y0_test = (y_test == '0')


# In[11]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y0_train)


# In[12]:


y_train_pred = sgd_clf.predict(X_train)
y_test_pred = sgd_clf.predict(X_test)


# In[13]:


from sklearn.metrics import accuracy_score
y_train_acc = accuracy_score(y0_train, y_train_pred)
y_test_acc = accuracy_score(y0_test, y_test_pred)


# In[14]:


import pickle
acc = [y_train_acc, y_test_acc]
with open('sgd_acc.pkl', 'wb') as acc_pickle:
    pickle.dump(acc, acc_pickle)


# In[15]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(sgd_clf, X_train, y0_train, cv=3, scoring='accuracy', n_jobs=-1)


# In[16]:


with open('sgd_cva.pkl', 'wb') as cva_pickle:
    pickle.dump(score, cva_pickle)


# In[17]:


sgd_clf_all = SGDClassifier(random_state=42)


# In[18]:


sgd_clf.fit(X_train, y_train)


# In[19]:


y_test_pred_all = sgd_clf.predict(X_test)


# In[20]:


accuracy_score(y_test, y_test_pred_all)


# In[21]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_test_pred_all)


# In[22]:


conf_matrix


# In[23]:


with open('sgd_cmx.pkl', 'wb') as cmx_pickle:
    pickle.dump(conf_matrix, cmx_pickle)


# In[ ]:




