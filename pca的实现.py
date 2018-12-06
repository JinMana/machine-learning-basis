#!/usr/bin/env python
# coding: utf-8
降维是对数据高维度特征的一种预处理方法。降维是将高维度的数据保留下最重要的一些特征，去除噪声和不重要的特征，从而实现提升数据处理速度的目的。在实际的生产和应用中，降维在一定的信息损失范围内，可以为我们节省大量的时间和成本。降维也成为了应用非常广泛的数据预处理方法。

　　降维具有如下一些优点：

（1）使得数据集更易使用

（2）降低算法的计算开销

（3）去除噪声

（4）使得结果容易理解

PCA(principal Component Analysis)，即主成分分析方法，是一种使用最广泛的数据压缩算法。在PCA中，数据从原来的坐标系转换到新的坐标系，由数据本身决定。转换坐标系时，以方差最大的方向作为坐标轴方向
协方差是一种用来度量两个随机变量关系的统计量通过计算数据矩阵的协方差矩阵，然后得到协方差矩阵的特征值及特征向量，选择特征值最大（也即包含方差最大）的N个特征所对应的特征向量组成的矩阵，我们就可以将数据矩阵转换到新的空间当中，实现数据特征的降维（N维）。
方差是一维的，协方差至少两维
# In[13]:


import numpy as np
import pandas as pd


# In[14]:


def load_data(filename):
    #按空格分开
    dataMat = pd.read_csv(filename, header=None, sep='\t')
    #111
    return dataMat


# In[15]:


def pca(dataMat, topNfeat=9999):
    '''
    dataMat:输入的数据
    topNfeat:保留的特征值
    '''
    #获取每一列的均值
    dataMat = np.mat(dataMat)
    mean_val = np.mean(dataMat, axis=0)
    mean_val= np.mat(mean_val)
    
    #每个特征减去均值
    data_rm_mean = dataMat - mean_val
    
    #计算协方差矩阵
    covMat = np.cov(data_rm_mean, rowvar=0)
    #计算协方差举证对应的特征值和特征向量
    eig_val, eig_vect = np.linalg.eig(np.mat(covMat))
    
    #特征值排序，然后返回下标
    eig_index = np.argsort(eig_val)
    #自下而上选取n个值
    eig_index = eig_index[:-(topNfeat+1):-1]
    #组成压缩矩阵，即特征向量
    redEigVec = eig_vect[:, eig_index]
    #协方差举证*压缩矩阵,转换为新的空间，维度变为N
    lowMat = (data_rm_mean * redEigVec )  #DX
    #反构原数据
    reMat = (lowMat * redEigVec.T) + mean_val  #DD.TX
    return lowMat, reMat


# In[16]:


import matplotlib
import matplotlib.pyplot as plt


# In[17]:


dataMat = load_data("testSet.txt")


# In[19]:


lowMat, reMat = pca(dataMat, 1)  #1维
# print(lowMat.shape)


# In[40]:


dataMat = np.array(dataMat)
lowMat = np.array(lowMat)
reMat = np.array(reMat)


# In[43]:


fig = plt.figure()
ax = fig.add_subplot(111)
#原数据三角，而降维后数据是圆形
ax.scatter(dataMat[:,0],dataMat[:, 1], marker='^', s=10)         #原数据
ax.scatter(reMat[:,0], reMat[:,1], marker='o',s=10, c='red')     #降维复原后的数据
plt.show()

