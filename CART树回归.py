#!/usr/bin/env python
# coding: utf-8

# CART是用二元切分来处理连续型特征
1、预剪枝，也是前剪枝，通过设置阈值、节点数、误差大小来控制枝的多少
  缺点：需要提前设置阈值
# In[1]:


import numpy as np
import pandas as pd


# In[56]:


def load_data():
    dataMat = []
    dataLabel = []
    
    #按空格分开
    dataMat = pd.read_csv("ex0.txt", header=None, sep="\s+")
    #111
    return dataMat


# In[77]:


def binsplitdata(dataset, feature, value):
    '''
    某个特征 》 value 就在mat0
    某个特征 《 value 就在mat1
    '''
    #np.nonzero获取非0的下标
    
    #这里不进行转换时，最后一次迭代的dataset的类型会自动变成list类型
    dataset = np.mat(dataset)  
    
    #mat0, mat1 = binsplitdata(dataMat, 1, 0.44181499999999996)
    mat0 = dataset[np.nonzero(dataset[:, feature] > value)[0], :]
    mat1 = dataset[np.nonzero(dataset[:, feature] <= value)[0], :]
    return mat0, mat1


# In[78]:


def regLeaf(dataset):
    return np.mean(dataset[:,-1])

def regErr(dataset):
    return np.var(dataset[:,-1])* dataset.shape[0]   #标签方差*m

def choosebestsplit(dataset, leafType, errType, ops):
    '''
    用于获取特征和阈值
    ops是规定了每个分支的最多的节点
    '''
    tolN = ops[1]    #每个分支最多有4个节点
    tolS = ops[0]   #误差容忍值
    
    dataset = np.mat(dataset)
    #np.set(dataset[:, -1].T.tolist()[0])获取label的唯一值，如果都是1，那么就退出
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataset)   #求dataset均值
    
    S = errType(dataset)   #结果方差之和
    
    #最好的方差初始化
    bestS = np.float('inf')
    
    best_index = 0
    best_value = 0
    
    m, n = dataset.shape
    #获取每个特征
    for i in range(n - 1):          #最后一个label
        #val是每个特征的所有值中的一个
        for val in set(dataset[:, i].flat):
            mat0, mat1 = binsplitdata(dataset, i, val)
            if(mat0.shape[0]  < tolN) or (mat1.shape[0] < tolN):
                continue
            #计算切分后的误差
            newS = errType(mat0) + errType(mat1)
            
            if newS < bestS:
                #最小损失更新
                bestS = newS
                best_index = i
                best_value = val
#     print("00000000000000")            
#     print(bestS)
#     print(S-bestS)
    if(S - bestS) < tolS:
        #切分前后的误差小于tolS，不需要切分，直接作为叶子节点
        return None, leafType(dataset)
    
#     print("kkkkkkkkkkkkkk")
#     print(best_index)
#     print(best_value)
    #用1 和0.44181499999999996再分
    mat0, mat1 = binsplitdata(dataset, best_index, best_value)
#     print(mat0.shape[0])
#     print(mat1.shape[0])
    
    if(mat0.shape[0] < tolN)  or (mat1.shape[0] < tolN):
        #不满足最小的切分条件
        return None, leafType(dataset)
    return best_index, best_value


# In[79]:


def createTree(dataset, leafType=regLeaf, errType=regErr, ops=(1,4)):
    '''
    创建树
    '''
    #choosebestsplit获取最好的特征，和阈值
    feature, val = choosebestsplit(dataset, leafType, errType, ops)
    
    if feature == None:
        return val

    retTree = {}
    retTree['feature'] = feature
    retTree['value'] = val
    
    #用binsplitdata分为左右两个子树
    
    #这里出问题了
    #mat0, mat1 = binsplitdata(dataset, 1, 0.44181499999999996)
    mat0, mat1 = binsplitdata(dataset, feature, val)
    
    #用迭代的方式，使左右子树进行迭代
    retTree['left'] = createTree(mat0, leafType, errType, ops)
    retTree['right'] = createTree(mat1, leafType, errType, ops)

    return retTree


# In[80]:


dataMat = load_data()
#将datafrme变成list集合
dataMat =  dataMat.values.tolist()
tree = createTree(dataMat)


# In[81]:


tree

2、后剪枝
 需要训练集和测试集，构建的树要大且复杂，便于后面的剪枝
 通过节点合并后的误差和不合并的误差
# In[82]:


def istree(obj):
    '''
    是否是字典类型，是就返回true，不是就返回false
    '''
    return (type(obj).__name__ == 'dict')


# In[83]:


def getmean(tree):
    if istree(tree['right']):
        #递归获取均值
        tree['right'] = getmean(tree['right'])
    if istree(tree['left']):
        tree['left'] = getmean(tree['left'])
    
    m = (tree['right'] + tree['left']) / 2.0
    return m


# In[86]:


def prune(tree,testData):
    #测试集为空，直接对树相邻叶子结点进行求均值操作
    if shape(testData)[0]==0:
        return getMean(tree)
    
    #左右分支中有非叶子结点类型
    if (isTree(tree['right']) or isTree(tree['left'])):
        #利用当前树的最佳切分点和特征值对测试集进行树构建过程
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spval'])
        
    #左分支非叶子结点，递归利用测试数据的左子集对做分支剪枝
    if isTree(tree['left']):
        tree['left']=prune(tree['left'],lSet)
        
    #同理，右分支非叶子结点，递归利用测试数据的右子集对做分支剪枝
    if isTree(tree['right']):
        tree['right']=prune(tree['right'],lSet)
        
    #左右分支都是叶节点
    if not isTree(tree['left']) and not isTree(tree['right']):
        #利用该子树对应的切分点对测试数据进行切分(树构建)
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spval'])
        
        #如果这两个叶节点不合并，计算误差，即（实际值-预测值）的平方和
        errorNoMerge=sum(np.power(lSet[:,-1]-tree['left'],2)) + sum(np.power(rSet[:,-1]-tree['right'],2))
        #求两个叶结点值的均值
        treeMean=(tree['left']+tree['right'])/ 2.0
        #如果两个叶节点合并，计算合并后误差,即(真实值-合并后值）平方和
        errorMerge=sum(power(testData[:,-1]-treeMean,2))
       
    #合并后误差小于合并前误差
        if errorMerge<errorNoMerge:
            #和并两个叶节点，返回合并后节点值
            print('merging')
            return treeMean
        #否则不合并，返回该子树
        else:return tree
    #不合并，直接返回树
    else:return tree


# In[ ]:


prune(tree, )


# In[ ]:




