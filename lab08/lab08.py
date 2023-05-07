# %%
from sklearn import datasets
import pandas as pd
data_breast_cancer = datasets.load_breast_cancer()
df_breast_cancer = pd.DataFrame(data_breast_cancer.data, columns=data_breast_cancer.feature_names)
df_breast_cancer = df_breast_cancer.assign(target=pd.Series(data_breast_cancer.target))
df_breast_cancer.head()

# %%
from sklearn.datasets import load_iris
data_iris = load_iris()
df_iris = pd.DataFrame(data_iris.data, columns=data_iris.feature_names)
df_iris = df_iris.assign(target=pd.Series(data_iris.target))
df_iris.head()

# %%
from sklearn.decomposition import PCA
import numpy as np
pca_breast_cancer = PCA(n_components=0.9)
breast_cancer_reduced = pca_breast_cancer.fit_transform(df_breast_cancer)
print(pca_breast_cancer.explained_variance_ratio_)
print(df_breast_cancer.shape, breast_cancer_reduced.shape)
print(df_breast_cancer.columns[np.argmax(abs(pca_breast_cancer.components_[0]))])

# %%
pca_iris = PCA(n_components=0.9)
iris_reduced = pca_iris.fit_transform(df_iris)
print(pca_iris.explained_variance_ratio_)
print(df_iris.shape, iris_reduced.shape)
print(df_iris.columns[np.argmax(abs(pca_iris.components_[0]))])

# %%
# Scaled data
from sklearn.preprocessing import StandardScaler
scaler_breast_cancer = StandardScaler()
scaler_breast_cancer.fit(df_breast_cancer)
scaled_breast_cancer = scaler_breast_cancer.transform(df_breast_cancer)
pca_breast_cancer = PCA(n_components=0.9)
breast_cancer_reduced = pca_breast_cancer.fit_transform(scaled_breast_cancer)
print(pca_breast_cancer.explained_variance_ratio_)
breast_cancer_variance_list = list(pca_breast_cancer.explained_variance_ratio_)
print(df_breast_cancer.shape, breast_cancer_reduced.shape)
breast_cancer_components_list = []
for i in pca_breast_cancer.components_:
    breast_cancer_components_list.append(np.argmax(abs(i)))
    print(df_breast_cancer.columns[np.argmax(abs(i))])
print(breast_cancer_components_list)

# %%
scaler_irirs = StandardScaler()
scaler_irirs.fit(df_iris)
scaled_iris = scaler_irirs.transform(df_iris)
pca_iris = PCA(n_components=0.9)
iris_reduced = pca_iris.fit_transform(scaled_iris)
print(pca_iris.explained_variance_ratio_)
iris_variance_list = list(pca_iris.explained_variance_ratio_)
print(df_iris.shape, iris_reduced.shape)
iris_components_list = []
for i in pca_iris.components_:
    iris_components_list.append(np.argmax(abs(i)))
    print(df_iris.columns[np.argmax(abs(i))])
print(iris_components_list)

# %%
import pickle

with open('pca_bc.pkl', 'wb') as fb:
    pickle.dump(breast_cancer_variance_list, fb)

with open('pca_ir.pkl', 'wb') as fi:
    pickle.dump(iris_variance_list, fi)

# %%
with open('idx_bc.pkl', 'wb') as fb_compontents:
    pickle.dump(breast_cancer_components_list, fb_compontents)

with open('idx_ir.pkl', 'wb') as fi_compontents:
    pickle.dump(iris_components_list, fi_compontents)


