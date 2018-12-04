#!/usr/bin/env python
# coding: utf-8
主要介绍PCA的使用

PCA作为一种无监督数据压缩算法，只保留最重要的主方向，则在压缩时，自变量和因变量间的关系有可能变的更复杂了。

PCA在特征之间有大相关性时效果通常不错。
# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


from sklearn.datasets.samples_generator import make_blobs


# In[3]:


#数据集
x, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3,3],[0,0,0],[1,1,1],[2,2,2]],cluster_std=[0.2, 0.1, 0.2, 0.2],random_state=9)


# In[19]:


x.shape


# In[4]:


#画图
fig = plt.figure()
ax = Axes3D(fig, rect=[0,0,1,1],elev=30, azim=20)
plt.scatter(x[:,0], x[:,1], x[:,2], marker='o')


# In[5]:


#不降维，投影后的三个维度的方差分布
from sklearn.decomposition import PCA


# In[7]:


pca = PCA(n_components=3)
pca.fit(x)
print(pca.explained_variance_ratio_)   #方差占比
print(pca.explained_variance_)         #方差

[0.98318212 0.00850037 0.00831751，可以发现]
[3.78521638 0.03272613 0.03202212]
三个特征（即三维）的比例是0.98318212 0.00850037 0.00831751，可以发现第一个特征占主要成分
# In[8]:


#三维降二维
pca = PCA(n_components=2)
pca.fit(x)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)


# In[13]:


#查看转换后的数据分布,可以清晰的看出是四个簇
x_new = pca.transform(x)
plt.scatter(x_new[:,0], x_new[:, 1], marker="o")
plt.show()


# In[14]:


#不指定维度而指定方差
#指定了主成成分至少95%
pca = PCA(n_components=0.95)
pca.fit(x)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)


# In[16]:


#第一个特征和第二个特征的和可以满足99
pca = PCA(n_components=0.99)
pca.fit(x)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)


# In[17]:


#用mle算法降维
pca = PCA(n_components='mle')
pca.fit(x)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)


# In[24]:


from sklearn.decomposition import MiniBatchSparsePCA, KernelPCA


# In[20]:


#MiniBatchSparsePCA
min_pca = MiniBatchSparsePCA(n_components=2, batch_size=1000, normalize_components=True, random_state=0)


# In[23]:


k = min_pca.fit_transform(x)
k.shape


# In[25]:


#MiniBatchSparsePCA
kpca = KernelPCA(n_components=2, kernel='linear')
kpca.fit_transform(x)

