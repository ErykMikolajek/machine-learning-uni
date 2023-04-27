# %%
from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto') 
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]

# %%
from sklearn.cluster import KMeans

kmeans = []
predictions = []
for k in range(8, 13):
    kmeans.append(KMeans(n_clusters=k, random_state=42))
    predictions.append(kmeans[-1].fit_predict(X))

# %%
from sklearn.metrics import silhouette_score

silhouettes = []
for k in range(8, 13):
    silhouettes.append(silhouette_score(X, kmeans[k-8].labels_))

# %%
import pickle

with open('kmeans_sil.pkl', 'wb') as sil_pickle:
    pickle.dump(silhouettes, sil_pickle)

# %%
from sklearn.metrics import confusion_matrix

kmeans_10 = kmeans[2]
y_pred = predictions[2]

matrix = confusion_matrix(y, y_pred)
matrix

# %%
max_index = set(np.argmax(matrix, axis=1))
max_index = list(max_index)

# %%
with open('kmeans_argmax.pkl', 'wb') as args_pickle:
    pickle.dump(max_index, args_pickle)

# %%
print(X)
X.shape[0]

# %%
distances = np.zeros((300,X.shape[0]))
for i in range(300):
    for j in range(X.shape[0]):
        dist = np.linalg.norm(X[i] - X[j])
        if dist != 0:
            distances[i, j] = dist
        else:
            distances[i, j] = np.inf

# %%
smallest_distances = sorted(distances.ravel())[:10]
print(smallest_distances)

# %%
with open('dist.pkl', 'wb') as dist_pickle:
    pickle.dump(smallest_distances, dist_pickle)

# %%
smallest_3 = smallest_distances[:3]
s = np.mean(smallest_3)

# %%
from sklearn.cluster import DBSCAN
dbscan_len = []
step = s
while step < s + 0.10 * s:
    dbscan = DBSCAN(eps=step)
    dbscan.fit(X)
    dbscan_len.append(len(set(dbscan.labels_)))
    step += 0.04 * s

# %%
with open('dbscan_len.pkl', 'wb') as dbscan_pickle:
    pickle.dump(dbscan_len, dbscan_pickle)

# %%



