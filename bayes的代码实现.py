#!/usr/bin/env python
# coding: utf-8
朴素贝叶斯算法，显然除了贝叶斯准备，朴素一词同样重要。这就是我们要说的条件独立性假设的概念。条件独立性假设是指特征之间的相互
独立性假设，所谓独立，是指的是统计意义上的独立，即一个特征或者单词出现的可能性与它和其他单词相邻没有关系。文档特征的方法。一种是词集模型，另外一种是词袋模型。顾名思义，词集模型就是对于一篇文档中出现的每个词，我们不考虑其
出现的次数，而只考虑其在文档中是否出现，并将此作为特征；假设我们已经得到了所有文档中出现的词汇列表，那么根据每个词
是否出现，就可以将文档转为一个与词汇列表等长的向量。而词袋模型，就是在词集模型的基础上，还要考虑单词在文档中出现的次数
，从而考虑文档中某些单词出现多次所包含的信息。
# In[1]:


import numpy as np
import math


# In[2]:


def create_dataset():
    '''
    file_list中每一行表示每一个文档，并且每一行的大小不同
    '''
    file_list = [['my','dog','has','flea','problems','help','please'],
                ['maybe','not','take','him','to','dog','park','stupid'],
                ['my','dalmation','is','so','cute','I','love','him'],
                ['stop','posting','stupid','worthless','garbage'],
                ['my','licks','ate','my','steak','how','to','stop','him'],
                ['quit','buying','worthless','dog','food','stupid']]
    #类标签
    file_labels = [0, 1, 0, 1, 0, 1]
    return file_list, file_labels


# In[3]:


def createVocalList(file_list):
    '''
    求文档中的词条，即出现过的词
    '''
    vocal_list = set([])
    for doc in file_list:
        vocal_list = vocal_list|set(doc)  #求两个set集合的并集
    return list(vocal_list)


# In[4]:


def setword(vocal_list, inputset):
    '''
    判断文档中是否出现过该词，1指出现过，0没有出现过
    '''
    returnvec = len(vocal_list)*[0]
    for vec in inputset:
        if vec in vocal_list:
            #出现过就是1
            returnvec[vocal_list.index(vec)] = 1
    return returnvec


# In[5]:


#vocal_list是词条列表
file_list, file_labels = create_dataset()
vocal_list = createVocalList(file_list)


# In[6]:


len(vocal_list)


# In[7]:


#词条向量，及文档中出现就是1，没有出现就是0,每个文档都有一个词条向量
vec = setword(vocal_list, file_list[0])
trainMatrix = []
for i in range(len(file_list)):
    vec = setword(vocal_list, file_list[i])
    trainMatrix.append(vec)
#每篇文档的vec
#类标签，及file_labels


# In[19]:


def tranNB(trainMatrix, trainCategory):
    '''
    trainMatrix表示一整个list中每个文档出现的的词条
    trainCategory每一篇文档的类标签
    
    '''
    #num_doc表示文档的数目，例如6
    num_doc = len(trainMatrix)
    #num_wor表示每个词条的长度，例如31
    num_wor = len(trainMatrix[0])
    #文档中类1所占的比例p(c=1)
    pc = sum(trainCategory)/float(num_doc)    #p（c=1）
    
#     p0_num = np.zeros(num_wor)
#     p1_num = np.zeros(num_wor)
    
#     p0demo = 0.0
#     p1demo = 0.0
    
    #算法改进部分，当p(w0/c)*p(w1/c)...有一个为0的时候，那整个概率为0，为了避免情况，将分子初始化为1，分母初始化为2
    p0_num = np.ones(num_wor)
    p1_num = np.ones(num_wor)
    
    p0demo = 2.0
    p1demo = 2.0
    
    #遍历每一个文档
    for i in range(num_doc):
        #判断文档类别是否为1
        if(trainCategory[i] == 1):
            #类别为1的文档,放在p1_num里面 
            p1_num += trainMatrix[i]      #在侮辱性词语中，某个词出现的次数
            #计算词条总数
            p1demo += sum(trainMatrix[i])    #所有侮辱性的词语
        else:
            p0_num += trainMatrix[i]
            p0demo += sum(trainMatrix[i])
    #类别为1的条数 和 词的数目 的商
#     p1vect = p1_num / p1demo      #p(w/c) 
#     p0vect = p0_num / p0demo
    
    #算法改进部分2
    #为了避免数据下溢，可以用对数
    p1vect = np.log(p1_num/p1demo)
    p0vect = np.log(p0_num/p0demo)
    
    #所以返回的是p(w/c=0) p(w/c=1) p(c=1)而p(c=0)可以用1-p(c=1)
    return p0vect, p1vect, pc


# In[20]:


tranNB(trainMatrix, file_labels)


# In[21]:


def classifyNB(vecClassify, p0vect, p1vect, pc):
    '''
    vecClassify:待分类的词条
    p0vect:类别0的文档中词条出现的频数p(w0|c0)
    p1vect:类别0的文档中词条出现的频数p(w1|c1)
    pc:类别为1的文档的比例
    '''
    p1 = sum(vecClassify*p1vect) + math.log(pc)   #这是什么
    p0 = sum(vecClassify*p0vect) + math.log(1.0 - pc)
    if p1 > p0:
        return 1
    else:
        return 0


# In[22]:


def testNB():
    #文档和标签
    file_list, file_labels = create_dataset()
    #词条
    vocal_list = createVocalList(file_list)
    #词条矩阵
    train_mat = []
    for doc in file_list:
        doc_vocal = setword(vocal_list, doc)
        train_mat.append(doc_vocal)
    p0vect, p1vect, pc = tranNB(np.array(train_mat), np.array(file_labels))
    
    #测试文档
    testfile = ['love','my','dalmation']
    #将测试文档转为词条
    test_doc_vocal = np.array(setword(vocal_list, testfile))
    print(testfile, "is", classifyNB(test_doc_vocal,p0vect, p1vect, pc))
    
    #测试文档2
    testfile2 = ['stupid','garbage']
    test_doc_vocal2 = np.array(setword(vocal_list, testfile2))
    print(testfile2,"is", classifyNB(test_doc_vocal2, p0vect, p1vect, pc))


# In[23]:


testNB()

