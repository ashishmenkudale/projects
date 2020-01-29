
# coding: utf-8

# In[14]:


from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


X = random_walks(n_ts=50, sz=32, d=1, random_state= 0)


# In[9]:


km5 = TimeSeriesKMeans(n_clusters=5, metric="euclidean", max_iter=5,random_state=0).fit(X)
km3 = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5,random_state=0).fit(X)


# In[10]:


km_dba3 = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5, max_iter_barycenter=5,random_state=0).fit(X)
km_dba4 = TimeSeriesKMeans(n_clusters=4, metric="dtw", max_iter=5, max_iter_barycenter=5,random_state=0).fit(X)
km_dba5 = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=5, max_iter_barycenter=5,random_state=0).fit(X)


# In[11]:


km_sdtw3 = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5,max_iter_barycenter=5,metric_params={"gamma": .5},random_state=0).fit(X)
km_sdtw4 = TimeSeriesKMeans(n_clusters=4, metric="softdtw", max_iter=5,max_iter_barycenter=5,metric_params={"gamma": .5},random_state=0).fit(X)
km_sdtw5 = TimeSeriesKMeans(n_clusters=5, metric="softdtw", max_iter=5,max_iter_barycenter=5,metric_params={"gamma": .5},random_state=0).fit(X)


# In[12]:


km5_p = km5.predict(X)
km3_p = km3.predict(X)

km_dba3_p = km_dba3.predict(X)
km_dba4_p = km_dba4.predict(X)
km_dba5_p = km_dba5.predict(X)

km_sdtw3_p = km_sdtw3.predict(X)
km_sdtw4_p = km_sdtw4.predict(X)
km_sdtw5_p = km_sdtw5.predict(X)


# In[15]:


l0 = X[np.where(km5_p == 0)]
l1 = X[np.where(km5_p == 1)]
l2 = X[np.where(km5_p == 2)]
l3 = X[np.where(km5_p == 3)]
l4 = X[np.where(km5_p == 4)]


# In[24]:


pd.Panel(l0).to_frame().plot()
pd.Panel(l1).to_frame().plot()
pd.Panel(l2).to_frame().plot()
pd.Panel(l3).to_frame().plot()
pd.Panel(l4).to_frame().plot()


# In[19]:


pd.Panel(l1).to_frame().plot()


# In[20]:


pd.Panel(l2).to_frame().plot()


# In[21]:


pd.Panel(l3).to_frame().plot()


# In[22]:


pd.Panel(l4).to_frame().plot()


# In[ ]:


import numpy as np
import pandas as pd
l0 = X[np.where(km_predicted == 0)]
l1 = X[np.where(km_predicted == 1)]
l2 = X[np.where(km_predicted == 2)]
l3 = X[np.where(km_predicted == 3)]
l4 = X[np.where(km_predicted == 4)]


# In[ ]:





# In[ ]:


from matplotlib import pyplot
pd.Panel(l1).to_frame().plot()


# In[ ]:


from matplotlib import pyplot
pd.Panel(l2).to_frame().plot()


# In[ ]:


from matplotlib import pyplot
pd.Panel(l3).to_frame().plot()


# In[ ]:


from matplotlib import pyplot
pd.Panel(l4).to_frame().plot()


# In[ ]:


# Author: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance,     TimeSeriesResampler

seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = X_train[y_train < 4]  # Keep first 3 classes
numpy.random.shuffle(X_train)
# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])
# Make time series shorter
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
sz = X_train.shape[1]

# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=3, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

plt.figure()
for yi in range(3):
    plt.subplot(3, 3, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Euclidean $k$-means")

# DBA-k-means
print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=3,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
y_pred = dba_km.fit_predict(X_train)

for yi in range(3):
    plt.subplot(3, 3, 4 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("DBA $k$-means")

# Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=3,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
y_pred = sdtw_km.fit_predict(X_train)

for yi in range(3):
    plt.subplot(3, 3, 7 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()

