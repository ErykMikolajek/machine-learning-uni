#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y}) 
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')


# In[2]:


from sklearn.metrics import mean_squared_error
mse = {}


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train= X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


# In[4]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[5]:


mse['lin_reg'] = [mean_squared_error(lin_reg.predict(X_train), y_train), mean_squared_error(lin_reg.predict(X_test), y_test)]


# In[6]:


import matplotlib.pyplot as plt
plt.scatter(df.x, df.y)
x_lin_reg = np.linspace(-3,3,100)
y_lin_reg = lin_reg.coef_*x_lin_reg + lin_reg.intercept_
plt.plot(x_lin_reg, y_lin_reg)


# In[7]:


import sklearn.neighbors
knn3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn3_reg.fit(X_train, y_train)


# In[8]:


mse['knn_3_reg'] = [mean_squared_error(knn3_reg.predict(X_train), y_train), mean_squared_error(knn3_reg.predict(X_test), y_test)]


# In[9]:


plt.scatter(df.x, df.y)
x_knn3_reg = np.linspace(-3,3,100)
y_knn3_reg = knn3_reg.predict(x_knn3_reg.reshape(-1, 1))
plt.plot(x_knn3_reg, y_knn3_reg, color='red')


# In[10]:


knn5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn5_reg.fit(X_train, y_train)


# In[11]:


mse['knn_5_reg'] = [mean_squared_error(knn5_reg.predict(X_train), y_train), mean_squared_error(knn5_reg.predict(X_test), y_test)]


# In[12]:


plt.scatter(df.x, df.y)
x_knn5_reg = np.linspace(-3,3,100)
y_knn5_reg = knn3_reg.predict(x_knn5_reg.reshape(-1, 1))
plt.plot(x_knn5_reg, y_knn5_reg, color='red')


# In[13]:


from sklearn.preprocessing import PolynomialFeatures
poly_features2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly_features2.fit_transform(X_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly2, y_train)


# In[14]:


mse_X_train_poly = poly_features2.fit_transform(X_train)
mse_X_test_poly = poly_features2.fit_transform(X_test)
mse['poly_2_reg'] = [mean_squared_error(lin_reg2.predict(mse_X_train_poly), y_train), mean_squared_error(lin_reg2.predict(mse_X_test_poly), y_test)]


# In[15]:


plt.scatter(df.x, df.y)
x_lin_reg2 = np.linspace(-3,3,100)
y_lin_reg2 = lin_reg2.coef_[1] * x_lin_reg2**2 + lin_reg2.coef_[0] * x_lin_reg2 + lin_reg2.intercept_
plt.plot(x_lin_reg2, y_lin_reg2, color='red')


# In[16]:


poly_features3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly_features3.fit_transform(X_train)
lin_reg3 = LinearRegression()
lin_reg3.fit(X_poly3, y_train)


# In[17]:


mse_X_train_poly = poly_features3.fit_transform(X_train)
mse_X_test_poly = poly_features3.fit_transform(X_test)
mse['poly_3_reg'] = [mean_squared_error(lin_reg3.predict(mse_X_train_poly), y_train), mean_squared_error(lin_reg3.predict(mse_X_test_poly), y_test)]


# In[18]:


plt.scatter(df.x, df.y)
x_lin_reg3 = np.linspace(-3,3,100)
y_lin_reg3 = lin_reg3.coef_[2] * x_lin_reg3**3 + lin_reg3.coef_[1] * x_lin_reg3**2 + lin_reg3.coef_[0] * x_lin_reg3 + lin_reg3.intercept_
plt.plot(x_lin_reg3, y_lin_reg3, color='red')


# In[19]:


poly_features4 = PolynomialFeatures(degree=4, include_bias=False)
X_poly4 = poly_features4.fit_transform(X_train)
lin_reg4 = LinearRegression()
lin_reg4.fit(X_poly4, y_train)


# In[20]:


mse_X_train_poly = poly_features4.fit_transform(X_train)
mse_X_test_poly = poly_features4.fit_transform(X_test)
mse['poly_4_reg'] = [mean_squared_error(lin_reg4.predict(mse_X_train_poly), y_train), mean_squared_error(lin_reg4.predict(mse_X_test_poly), y_test)]


# In[21]:


plt.scatter(df.x, df.y)
x_lin_reg4 = np.linspace(-3,3,100)
y_lin_reg4 = lin_reg4.coef_[3] * x_lin_reg4**4 + lin_reg4.coef_[2] * x_lin_reg4**3 + lin_reg4.coef_[1] * x_lin_reg4**2 + lin_reg4.coef_[0] * x_lin_reg4 + lin_reg4.intercept_
plt.plot(x_lin_reg4, y_lin_reg4, color='red')


# In[22]:


poly_features5 = PolynomialFeatures(degree=5, include_bias=False)
X_poly5 = poly_features5.fit_transform(X_train)
lin_reg5 = LinearRegression()
lin_reg5.fit(X_poly5, y_train)


# In[23]:


mse_X_train_poly = poly_features5.fit_transform(X_train)
mse_X_test_poly = poly_features5.fit_transform(X_test)
mse['poly_5_reg'] = [mean_squared_error(lin_reg5.predict(mse_X_train_poly), y_train), mean_squared_error(lin_reg5.predict(mse_X_test_poly), y_test)]


# In[24]:


plt.scatter(df.x, df.y)
x_lin_reg5 = np.linspace(-3,3,100)
y_lin_reg5 = lin_reg5.coef_[4] * x_lin_reg5**5 + lin_reg5.coef_[3] * x_lin_reg5**4 + lin_reg5.coef_[2] * x_lin_reg5**3 + lin_reg5.coef_[1] * x_lin_reg5**2 + lin_reg5.coef_[0] * x_lin_reg5 + lin_reg5.intercept_
plt.plot(x_lin_reg5, y_lin_reg5, color='red')


# In[25]:


df_mse = pd.DataFrame.from_dict(mse, orient='index', columns=['train_mse', 'test_mse'])


# In[26]:


import pickle
with open('mse.pkl', 'wb') as mse_pickle:
    pickle.dump(df_mse, mse_pickle)


# In[27]:


regressors_list = [(lin_reg, None), (knn3_reg, None), (knn5_reg, None), (lin_reg2, poly_features2), (lin_reg3, poly_features3), (lin_reg4, poly_features4), (lin_reg5, poly_features5)]
with open('reg.pkl', 'wb') as reg_pickle:
    pickle.dump(regressors_list, reg_pickle)
