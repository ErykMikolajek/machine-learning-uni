#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets


# In[2]:


data_breast_cancer = datasets.load_breast_cancer()
print(data_breast_cancer['DESCR'])


# In[3]:


data_iris = datasets.load_iris()
print(data_iris['DESCR'])


# In[4]:


import pandas as pd
import numpy as np
cancer_df_X = pd.DataFrame(data=data_breast_cancer['data'], columns= data_breast_cancer['feature_names'])
cancer_df_y = pd.DataFrame(data=data_breast_cancer['target'], columns=['target'])


# In[5]:


cancer_df_X = cancer_df_X[['mean area', 'mean smoothness']]


# In[6]:


from sklearn.model_selection import train_test_split
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(cancer_df_X, cancer_df_y, test_size=0.2, random_state=42)


# In[7]:


from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
clf_svc = LinearSVC(C=1, loss='hinge', random_state=42)
svc_scaled = Pipeline([('scaler', StandardScaler()), ('linear_svc', LinearSVC(C=1, loss='hinge', random_state=42))])


# In[8]:


clf_svc.fit(X_cancer_train, y_cancer_train)
svc_scaled.fit(X_cancer_train, y_cancer_train)


# In[9]:


clf_svc_predicted_train = clf_svc.predict(X_cancer_train)
clf_svc_predicted_test = clf_svc.predict(X_cancer_test)
svc_scaled_predicted_train = svc_scaled.predict(X_cancer_train)
svc_scaled_predicted_test = svc_scaled.predict(X_cancer_test)


# In[10]:


from sklearn.metrics import accuracy_score
clf_svc_accuracy_train = accuracy_score(y_cancer_train, clf_svc_predicted_train)
clf_svc_accuracy_test = accuracy_score(y_cancer_test, clf_svc_predicted_test)
svc_scaled_accuracy_train = accuracy_score(y_cancer_train, svc_scaled_predicted_train)
svc_scaled_accuracy_test = accuracy_score(y_cancer_test, svc_scaled_predicted_test)


# In[11]:


pickle_list = [clf_svc_accuracy_train, clf_svc_accuracy_test, svc_scaled_accuracy_train, svc_scaled_accuracy_test]


# In[12]:


import pickle
with open('bc_acc.pkl', 'wb') as acc_pickle:
    pickle.dump(pickle_list, acc_pickle)


# In[13]:


iris_df_X = pd.DataFrame(data=data_iris['data'], columns= data_iris['feature_names'])
iris_df_y = pd.DataFrame(data=data_iris['target'], columns=['target'])


# In[14]:


iris_df_X = iris_df_X[['petal length (cm)', 'petal width (cm)']]


# In[15]:


X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(iris_df_X, iris_df_y, test_size=0.2, random_state=42)


# In[16]:


clf_svc_iris = LinearSVC(C=1, loss='hinge', random_state=42)
svc_scaled_iris = Pipeline([('scaler', StandardScaler()), ('linear_svc', LinearSVC(C=1, loss='hinge', random_state=42))])


# In[17]:


clf_svc_iris.fit(X_iris_train, y_iris_train)
svc_scaled_iris.fit(X_iris_train, y_iris_train)


# In[18]:


iris_clf_svc_predicted_train = clf_svc_iris.predict(X_iris_train)
iris_clf_svc_predicted_test = clf_svc_iris.predict(X_iris_test)
iris_svc_scaled_predicted_train = svc_scaled_iris.predict(X_iris_train)
iris_svc_scaled_predicted_test = svc_scaled_iris.predict(X_iris_test)


# In[19]:


iris_clf_svc_accuracy_train = accuracy_score(y_iris_train, iris_clf_svc_predicted_train)
iris_clf_svc_accuracy_test = accuracy_score(y_iris_test, iris_clf_svc_predicted_test)
iris_svc_scaled_accuracy_train = accuracy_score(y_iris_train, iris_svc_scaled_predicted_train)
iris_svc_scaled_accuracy_test = accuracy_score(y_iris_test, iris_svc_scaled_predicted_test)


# In[20]:


pickle_list_iris = [iris_clf_svc_accuracy_train, iris_clf_svc_accuracy_test, iris_svc_scaled_accuracy_train, iris_svc_scaled_accuracy_test]


# In[21]:


import pickle
with open('iris_acc.pkl', 'wb') as acc_pickle_iris:
    pickle.dump(pickle_list_iris, acc_pickle_iris)

