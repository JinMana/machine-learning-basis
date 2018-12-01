#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


def load_dataset():
    dataMat = np.mat([[1. ,2.1],
        [2. ,1.1],
        [1.3,1. ],
        [1. ,1. ],
        [2. ,1. ]])
    classLabel = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat, classLabel


# In[3]:


dataMat, classLabel = load_dataset()


# In[4]:


dataMat


# In[5]:


classLabel


# In[110]:


def stumpClassify(dataMat, dimen, thre_val, thre_eq):
    '''
    单层决策树的过滤函数
    '''
#     print(dataMat)
    m, n = dataMat.shape
    retArray = np.ones((m, 1))   #初步设置都是分类正确的
    
    #决策树小于的分支
    if(thre_eq == "lt"):
        #小于的分支里面通过阈值设置不同的分支
        retArray[dataMat[:, dimen] < thre_val]  = -1.0
#         retArray[dataMat[0,dimen] <= thre_val] = -1.0
        
    #决策树大于的分支
    else:
        retArray[dataMat[:, dimen] > thre_val]  = -1.0
#         retArray[dataMat[0,dimen] > thre_val] = -1.0
    return retArray


# In[111]:


a = np.mat([[1. , 2.1],
 [2.  ,1.1],
 [1.3 ,1. ],
 [1. , 1. ],
 [2.,  1. ]])


# In[112]:


a[:,1]


# In[116]:


def buildStump(dataMat, classLabel, D):
    '''
    D是指数据集之前w
    '''
    dataMat = np.mat(dataMat)
    classLabel = np.mat(classLabel).T
    m, n = dataMat.shape
    
    #步长区间的总数,分成10个空间
    num_step = 10.0
    #最优决策树信息
    best_Tree = {}
    #单层决策树的预测结果
    bestClassEst = np.mat(np.zeros((m, 1)))
    
    #最小错误率初始化为无穷，用于后面错误率的比较
    min_error = np.float("inf")
    
    #三个循环，1、遍历特征值 2、步长  3、大于或小于
    for i in range(n):
        #该特征中的最大值和最小值
        range_min = dataMat[:,i].min()
        range_max = dataMat[:,i].max()
        
        #计算每个步长空间的大小
        step_size = (range_max-range_min) / num_step
        for j in range(-1, np.int(num_step)+1):     #比总的区间大没有关系吗
            #两种阈值的过滤形式
            for inequal in ["lt", "gt"]:
                #阈值计算
                thre_val = range_min + np.float(j)*step_size
                #调用上面那个函数进行预测
                predict_val = stumpClassify(dataMat, i, thre_val, inequal)
                
                #初始化错误向量
                errArray = np.ones((m, 1))
                #分类错误的是1，分类正确的是0
                errArray[predict_val == classLabel] = 0
                
                #计算加权错误率,即正确分类的样本点权重为0，错误分类的样本点权重为下面
                weight_error = D.T*errArray  #(1,5),(5,1) = (1,1)
                
#                 print("split: dim %d, thresh %.2f,thresh inequal: %s, the weighted error is %.3f" %(i,thre_val,inequal,weight_error))
                
                #如果错误率小于当前最小的错误率，把最小的错误率设为当前d 
                if(weight_error[0,0] < min_error):
                    min_error = weight_error[0,0]
                    bestClassEst = predict_val.copy()
                    best_Tree['dim'] = i
                    best_Tree['thre_val'] = thre_val
                    best_Tree['ineq'] = inequal
    
    #返回最佳单决策树的字典、错误率和预测输出结果             
    return best_Tree, min_error, bestClassEst


# In[114]:


dataMat, classLabel = load_dataset()
D = np.mat(np.ones((5,1))/5)
buildStump(dataMat, classLabel, D)


# In[117]:


def adaBoost(dataMat, classLabel, numIt = 40):
    '''
    adaboost算法
    '''
    #弱分类器信息表
    weakClassArr = []
    m, n = dataMat.shape
    #初始化权重
    D = np.mat(np.ones((m, 1))/m)
    #累计估计值向量
    aggClassEst = np.mat(np.zeros((m, 1)))
    
    #迭代次数
    for i in range(numIt):
        buildStump(dataMat, classLabel, D)
        bestStump,error, classEst = buildStump(dataMat, classLabel, D)
        print(i)
        print(classEst.T)
        #求系数alpha
        alpha = np.float(0.5*math.log((1.0-error)/(np.max((error, 1e-16)))))
        bestStump["alpha"] = alpha
        #将决策树存入到弱分类器中
        weakClassArr.append(bestStump)
        #更新权值D
        expon = np.multiply(-1*alpha*np.mat(classLabel).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/sum(D)
        #累加当前单层决策树的加权预测值
        aggClassEst += alpha *classEst
        
        #求出错误分类的样本点个数
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabel).T, np.ones((m, 1)))
        
        #计算错误率
        rate = sum(aggErrors)/m
        
        if rate == 0.0:
            break
    return weakClassArr


# In[118]:


dataMat, classLabel = load_dataset()
weakclassarr = adaBoost(dataMat, classLabel, 12)

结果预测
# In[119]:


def adaclassify(datatoclass,classarr):
    datamat = np.mat(datatoclass)
    m = datamat.shape[0]
    aggre = np.mat(np.zeros((m,1)))
    for i in range(len(datamat)):
        classest = stumpClassify(datamat, classarr[i]['dim'], classarr[i]['thre_val'], classarr[i]['ineq'])
        aggre += classarr[i]['alpha']*classest
        print(aggre)
    return np.sign(aggre)


# In[120]:


result = adaclassify([0,0], weakclassarr)


# In[121]:


result


# In[123]:


result = adaclassify([[5,5],[0,0]], weakclassarr)


# In[124]:


result

