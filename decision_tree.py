#!/usr/bin/env python
# coding: utf-8

# # 决策树

# In[1]:


import math
import operator


# In[2]:


def cal_shan(dataset):
    '''
    计算某个特征的熵
    '''
    data_len = len(dataset)
    kind_class = {}
    shan = 0.0
    #取dataset每一行
    for dataline in dataset:
        #取每一行最后一个数
        feature_one = dataline[-1] 
        kind_class[feature_one] = kind_class.get(feature_one, 0) + 1
        
    for key in kind_class:   #获取的是key值
        prob = float(kind_class[key]) / data_len
        shan -= prob * math.log(prob, 2)
    return shan


# In[3]:


def Create_dataset():
    dataset = [[1,1,'yes'], [1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    #是这两个feature的名字
    labels = ['no surfacing', 'flippers']
    return dataset, labels


# In[7]:


def split_dataset(dataset, axis, value):
    '''
    功能：去除数据集中axis那一列的值，返回剩下的值
    dataset:数据集
    axis:某一列，如0 1 2
    value:值，如0 1 
    '''
    retDataset = []
    for data in dataset:
        if(data[axis] == value):    #有的str类型，可以用==
            reduceFeature = data[:axis]    #这两行代码就是为了去掉axis这一列，而取他的左右两列
            reduceFeature.extend(data[axis+1:])
            retDataset.append(reduceFeature)
    return retDataset  


# In[8]:


split_dataset(dataset, 0, 1)


# In[9]:


dataset


# In[27]:


#选择最好的特征
def choose_best_method(dataset):
    '''
    选择信息增益最大的那个特征的下标
    '''
    data_len = len(dataset[0])-1
    #dataset的初始熵
    base_shan = cal_shan(dataset)
    #信息增益
    gain_shan = 0.0
    #新的熵
    new_shan = 0.0
    #信息增益最大
    best_gain_shan = 0.0
    #熵最大的那个特征
    best_feature = -1
    
    for i in range(data_len):
        #获取到dataset第一列的所有数，第二列的所有数
        #[1, 1, 1, 0, 0]
        #[1, 1, 0, 1, 1]
        #取该特征的所有数
        feature_list = [example[i] for example in dataset]
        #只能获取一行
#         feature_list = dataset[i]
        unique_val = set(feature_list)  #{0, 1}
        for value in unique_val:
            retDataset = split_dataset(dataset, i, value)
            prob = len(retDataset) / float(len(dataset))    #浮点数
            new_shan += prob * cal_shan(retDataset)
            
        gain_shan = base_shan - new_shan   #上面那一行和下面这一行是信息增益,刚才的错误是信息增益写错，写成new_shan - base_shan
        if(gain_shan > best_gain_shan):
            best_gain_shan = gain_shan
            best_feature = i
    return best_feature


# In[28]:


best_feature = choose_best_method(dataset)


# In[29]:


#best_feature就是最好的那个标签的下标
best_feature


# # 拥有多个特征的决策树

# In[30]:


#情况，所有的特征分完后该分支下的类标签仍然不唯一，这时候要采用多数表决函数
def majorityCnt(classList):
    '''
    分支下存在不同的标签，采用voting的方式
    '''
    #存放标签的字典
    class_count = {}
    #遍历该分支的所有标签
    for vote in classList:
        class_count[vote] = class_count.get(vote, 0) +1
    #对class_count进行降序排序
    #key=operator.itemgetter(1)获取列表第一个域的值
    #reverse=True降序
    sorted_class = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    #sorted_class[0][0]返回分支下类别最多的哪一个
    return sorted_class[0][0]


# In[31]:


classList = [example[-1] for example in dataset]


# In[32]:


classList.count(classList[0])


# In[33]:


dataset


# In[34]:


labels


# In[35]:


def createTree(dataset, labels):
    '''
    产生一棵树
    用字典表示
    '''
    #类标签
    classList = [example[-1] for example in dataset]
    #判断类标签是否相同,若相同，说明他们都属于同一个类
    if(classList.count(classList[0]) ==len(classList)):
        return classList[0]
    
    #遍历完所有的特征后，数据集中只有一个标签列,归属于同一个类
    if(len(dataset[0]) == 1):
        return majorityCnt(classList)
    
    #确定最优特征,返回下标
    best_feature_index = choose_best_method(dataset)
    best_feature_label = labels[best_feature_index]
    #储存数的信息
    myTree = {best_feature_label:{}}
    
    #复制类标签
    subLabels = labels[:]
    #删除标签中最优的标签
    del(subLabels[best_feature_index])
    
    #获取最优特征坐在的列
    best_feature_value = [example[best_feature_index] for example in dataset]
    unique_value = set(best_feature_value)  #也就是0或者1
    
    #遍历每一个特征的取值
    for value in unique_value:
#         递归
        myTree[best_feature_label][value] = createTree(split_dataset(dataset, best_feature_index, value), subLabels)
    return myTree


# In[41]:


mytree = createTree(dataset, labels)


# In[42]:


mytree


# # 测试

# In[47]:


def classify_tree(input_tree, featurelabels, test):
    '''
    输入样本点得出分类的情况
    input_tree：输入的树
    featurelabels：特征label
    test：测试样本
    '''
    #获取树中的第一个节点
    first_node = list(input_tree.keys())[0]
    second_tree = input_tree[first_node]
    #第一个特征对应的索引值
    feature_index = featurelabels.index(first_node)
    #遍历所有的取值
    for key in second_tree.keys():
        if(test[feature_index] == key):
            #判断是否还是字典类型
            if(type(second_tree[key]).__name__ == "dict"):
                #如果是字典类型,继续分支
                classLabel = classify_tree(second_tree[key], featurelabels, test)
            else:
                #只有叶子节点
                classLabel = second_tree[key]
    return classLabel


# In[48]:


mytree


# In[49]:


classify_tree(mytree, labels, [1,0])


# In[50]:


classify_tree(mytree, labels, [1,1])


# # 决策树的存储

# In[51]:


import pickle


# In[55]:


def storeTree(input_tree, filename):
    #打开文件
    file = open(filename, 'wb')
    #将决策树写进文件中
    pickle.dump(input_tree, file)
    #关闭文件
    file.close()


# # 加载决策树

# In[56]:


def load_tree(filename):
    file = open(filename, 'rb')
    return pickle.load(file)


# In[57]:


#test
storeTree(mytree, 'mytree')


# In[58]:


mt = load_tree('mytree')


# In[59]:


mt


# In[60]:


classify_tree(mt, labels, [1,1])


# In[37]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False
# In[45]:


dtree = DecisionTreeClassifier(criterion='entropy')
iris = load_iris()
data = iris.data
labels = iris.target
data, labels = shuffle(data, labels)
train_data = data[0:140]
test_data = data[140:-1]
train_labels = labels[0:140]
test_labels = labels[140:-1]


# In[46]:


dtree.fit(train_data, train_labels)


# In[47]:


pre_d = dtree.predict(test_data)


# In[48]:


from sklearn.metrics import classification_report


# In[49]:


clr = classification_report(pre_d, test_labels)


# In[50]:


print(clr)

 precision    recall  f1-score   support

           0       1.00      1.00      1.00         4
           1       0.75      1.00      0.86         3
           2       1.00      0.50      0.67         2
           
 precision    recall  f1-score   support

           0       1.00      1.00      1.00         3
           1       0.67      1.00      0.80         2
           2       1.00      0.75      0.86         4