#!/usr/bin/env python
# coding: utf-8
这个代码的维度出现了错乱
# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import datetime


# In[2]:


def load_data():
    dataMat = []
    dataLabel = []
    
    #按空格分开
    file = pd.read_csv("dataSet.txt", header=None, sep="\s+")
    b = np.ones((len(file),1))
    dataMat = np.hstack((b, np.array(file[[0,1]])))
    dataLabel = np.array(file[[2]])
    #111
    return dataMat, dataLabel


# In[3]:


def selectJ(i, m):
    '''
    优化alphaJ
    '''
    j = i
    if(j==i):
        j = int(random.uniform(0, m))
    return j


# In[4]:


def clipJ(aj,H, L):
    '''
    约束范围L<=alphaj <=H
    '''
    if aj>H:
        aj = H
    if L>aj:
        aj = L
    return aj


# In[159]:


def smoSimple(dataMat, dataLabel, C, toler, maxIter):
    '''
    简单的smo算法
    '''
    dataMat = np.mat(dataMat)
    dataLabel = np.mat(dataLabel).transpose()
    
    #初始化b=0
    b = 0
    m, n = dataMat.shape
    #建立一个m行1列的向量
    alphas = np.mat(np.ones((m, 1)))
    #迭代次数
    iterm = 0
    
    while(iterm < maxIter):
        #改变alpha的对数
        alphachange = 0
        #遍历样本
        for i in range(m):
            #计算svm的预测值,wx+b  ???
            print((dataMat*dataMat[i,:].T).shape)
            print((np.multiply(alphas, dataLabel).T).shape)
            fxi = np.multiply(alphas, dataLabel).T*(dataMat*dataMat[i,:].T)+b
            #计算误差
            Ei = fxi - dataLabel.transpose()
            #print(Ei.shape)   #(100, 1)
            #判断是否满足KKT条件,y(wx+b)>1 y(wx+b)>1-toler就满足
            if((sum(dataLabel[0,i]*Ei)<-toler) and (alphas[i]<C)) or ((sum(dataLabel[0,i]*Ei)>toler) and (alphas[i]>0)):
                j = selectJ(i, m)
                fxj = np.multiply(alphas, dataLabel).T * (dataMat*dataMat[j,:].T)+b
                Ej = fxj - dataLabel.transpose()
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                
                #求出相应的上下边界
                k = np.array(dataLabel)
                if(k[0][i] != k[0][j]):
                    #把ai，aj限制在0，C之间
                    L = np.max([0, (alphas[j] - alphas[i])[0,0]])  #比较这两个数的大小
                    H = np.min([C, C + (alphas[j] - alphas[j])[0,0]])
                else:
                    
                    L = np.max([0, (alphas[j]+alphas[i])[0,0]-C])
                    H = np.min([C, (alphas[j]-alphas[j])[0,0]])
                
                if L==H:
                    print("L==H")
                    continue
                eta = 2.0*dataMat[i, :]*dataMat[j,:].T - dataMat[i,:]*dataMat[i,:].T - dataMat[j,:]*dataMat[j,:].T
                
                if eta >= 0:
                    print("eta >= 0")
                    continue
                
#                 print(Ei-Ej)
                alphas[j] -= dataLabel[0,j] * (Ei-Ej) / eta[0,0]
                alphas[i] = clipJ(alphas[0,j], H, L)
                
                #如果改变后alphaj的变化不大，则退出本次循环
                if(np.abs(alphas[j] - alphaJold) < 0.00001):
                    print("j is moving enough")
                    continue
                #否则计算alphai的值
                alphas[i] += dataLabel[j]*dataLabel[i]*(alphaJold-alphas[j])
                
                #计算两个alpha下的b的值
                b1 = b - Ei - dataLabel[i]*(alphas[i] - alphaIold)*dataMat[i,:]*dataMat[i,:].T - dataLabel[j]*                     (dataLabel[j] - alphaJold)*dataMat[j, :]*dataMat[j,:].T
                b2 = b - Ej - dataLabel[i]*(alphas[i] - alphaIold)*dataMat[i,:]*dataMat[i,:].T - dataLabel[j]*                     (dataLabel[j] - alphaJold)*dataMat[j, :]*dataMat[j,:].T
                
                #如果0<alphai<c: b = b1
                #如果0<alphaj<c: b = b2
                #alphai, alphaj=0\c
                if(0<alphas[i]) and (C>alphas[i]):
                    b = b1
                elif(0<alphas[j]) and (C>alphas[j]):
                    b = b2
                else:
                    b = (b1+b2) / 2.0
                
                print("iterm: %d i: %d, paird changed %d" %(iterm, i, alphachange))
        if(alphachange == 0 ):
            iterm += 1
        else:
            iterm = 0
        print("iterm number :%d" %iterm)
    return b, alphas


# In[160]:


dataMat, dataLabel = load_data()
b, alphas = smoSimple(dataMat, dataLabel, 0.6, 0.001, 40)


# In[130]:


b = np.mat([[1,2],[3,4]])


# In[131]:


c = b/0.5


# In[132]:


c


# In[ ]:




