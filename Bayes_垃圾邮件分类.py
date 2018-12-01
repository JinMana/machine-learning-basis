#!/usr/bin/env python
# coding: utf-8

# In[83]:


import os
import jieba
import re
import numpy as np


# In[2]:


#获取数据集的每个目录
path = os.getcwd()
normal_path = os.path.join(path, 'data\\normal')
spam_path = os.path.join(path, 'data\\spam')
test_path = os.path.join(path, 'data\\test')

#获取目录中的每个文件
normal_dir = os.listdir(normal_path)
spam_dir = os.listdir(spam_path)
test_dir = os.listdir(test_path)


# In[94]:


#停用词处理
stop_list = []
stop_words_path = os.path.join(path, 'data\\中文停用词表.txt')
for line in open(stop_words_path):
    stop_list.append(line)


# In[97]:


#存放词频
word_dict = {}
word_list = []

for filename in normal_dir:
    data_path = os.path.join(path,'data\\normal')
    file_name = os.path.join(data_path, filename)
    #每个文档清空,word_list存放的是每个文档的唯一并且不在停词里面的词
    word_list.clear()
    
    for line in open(file_name):
        #正则表达式,去除非中文
        relu = re.compile(r"[^\u4e00-\u9fa5]")
        line = relu.sub("", line)
        
        #用jieba进行分词,中文分词
        content = list(jieba.cut(line))
        
        #遍历每一个词
        for word in content:
            #没有停用词
            if word not in stop_list and word.strip() != "" and word != None:
                if word not in word_list:
                    word_list.append(word)
                    
    #词典
    #将每个文档中的词放在词典中，并计算所有文档中词出现的次数
    for i in word_list:
        #dict中出现过就加1，没出现过就默认为1
        word_dict[i] = word_dict.get(i, 0)+1
        
normal_dict = word_dict.copy()


# In[98]:


#对spam数据的处理
spam_path = os.path.join(path, 'data\\spam')
word_dict.clear()

for filename in spam_dir:
    word_list.clear()
    
    #打开文件
    f = open(os.path.join(spam_path, filename))
    #获取文件每一行
    for line in f:
        #正则
        relu = re.compile(r"[^\u4e00-\u9fa5]")
        line = relu.sub("", line)
        #用jieba进行分词
        content = jieba.cut(line)
        for s in content:
            #进行判断
            if s not in stop_list and s.strip()!='' and s!=None:
                if s not in word_list:
                    word_list.append(s)
    for k in word_list:
        word_dict[k] = word_dict.get(k, 0) +1
spam_dict = word_dict.copy()


# In[106]:


normal_len = len(normal_dir)
spam_len = len(spam_dir)

def get_test_words(test_dict, spam_dict, normal_dict, normal_len, spam_len):
    word_prodict = {}
    for word, num in test_dict.items():
        if word in spam_dict.keys() and word in normal_dict.keys():
            pw_s = spam_dict[word] / spam_len
            pw_n = normal_dict[word]/normal_len
            ps_w = pw_s/ (pw_s+pw_n)
            word_prodict.setdefault(word, ps_w)

        if word in spam_dict.keys() and word not in normal_dict.keys():
            pw_s = spam_dict[word]/spam_len
            pw_n = 0.01
            ps_w = pw_s / (pw_s+pw_n)
            word_prodict.setdefault(word, ps_w)

        if word not in spam_dict.keys() and word in normal_dict.keys():
            pw_s = 0.01
            pw_n = normal_dict[word] / normal_len
            ps_w = pw_s / (pw_s+pw_n)
            word_prodict.setdefault(word, ps_w)

        if word not in spam_dict.keys() and word not in normal_dict.keys():
            word_prodict.setdefault(word, 0.47)
            
    sorted(word_prodict.items(), key=lambda d:d[1], reverse=True)[0:15]
    return word_prodict


# In[111]:


def calBayes(word_prolist, spam_dict, normal_dict):
    ps_w = 1
    ps_n = 1
    for word, pro in word_prolist.items():
        ps_w *= pro
        ps_n *= (1-pro)
    p = ps_w/ (ps_w+ps_n)
    return p


# In[136]:


def calAccuracy(test_result_dict):
    r = 0
    e = 0
    for i, ic in test_result_dict.items():
        if((int(i)<1000 and ic==0) or (int(i)>1000 and ic==1)):
            r+=1
        else:
            e+=1
    return(r/(r+e))


# In[139]:


#对test数据的处理
test_result_dict = {}
test_path  = os.path.join(path, 'data\\test')
for filename in test_dir:
    test_file = os.path.join(test_path , filename)
    
    word_list.clear()
    test_dict.clear()
    word_dict.clear()
    
    f = open(test_file)
    for line in f:
        relu = re.compile(r"[^\u4e00-\u9fa5]")
        line = relu.sub("", line)
        content = jieba.cut(line)
        for i in content:
            if i not in stop_list and i.strip() != '' and i!=None:
                if i not in word_list:
                    word_list.append(i)
    for t in word_list:
        word_dict[t] = word_dict.get(t, 0)+1

    test_dict = word_dict.copy() #这个文档的test_dict
    
    word_prolist = get_test_words(test_dict, spam_dict, normal_dict, normal_len, spam_len)
   
    
    p = calBayes(word_prolist, spam_dict, normal_dict)
    if p>0.9:
        test_result_dict.setdefault(filename, 1)
    else:
        test_result_dict.setdefault(filename, 0)

t = calAccuracy(test_result_dict)
print("准确率"+np.str(t))
for i, ic in test_result_dict.items():
    print(i+"/"+np.str(ic))

