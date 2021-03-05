import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
'''
（1）n_neighbors			int型参数；取离样本点p第k个最近的距离,k=n_neighbors
（2）algorithm='auto'     str参数；即内部采用什么算法实现，['auto', 'ball_tree', 'kd_tree', 'brute']。
                         'brute':暴力搜索；默认'auto':自动根据数据选择合适的算法。一般低维数据用kd_tree速度
                         快，用ball_tree相对较慢。超过20维的高维数据用kd_tree效果不佳，而ball_tree效果好。                         
（3）leaf_size            int参数；基于以上介绍的算法，此参数给出了kd_tree或者ball_tree叶节点规模，叶节点的
                         不同规模会影响数的构造和搜索速度，同样会影响储树的内存的大小。                        
（4）contamination		设置样本中异常点的比例,默认为 0.1
（5）metric				str参数或者距离度量对象；即怎样度量距离。默认是闵氏距离minkowski。
（6）p					int参数；就是以上闵氏距离各种不同的距离参数，默认为2，即欧氏距离。p=1代表曼哈顿距离等
（7）metric_params        距离度量函数的额外关键字参数，一般不用管，默认为None。
（8）n_jobs				int参数；并行任务数，默认为1表示一个线程，设置为-1表示使用所有CPU进行工作。可以指定为
                         其他数量的线程。   
fit(X)训练数据
model._decision_function(data) #返回data数据集的每个样本点的异常分数，是一个负数
model._predict(data) #返回data数据集的每个元素的样本点异常与否，正常是1，异常是-1
'''
x = np.loadtxt('data/194444.txt',dtype=int,delimiter='\t')
#k值=4使分类准确率最高


#训练模型
k_range = range(3,9)
k_scores = []
for k in k_range:
    model = LocalOutlierFactor(n_neighbors=k, contamination=0.18)
    model.fit(x)
    scores_pred = -model._decision_function(x)
    k_scores.append(scores_pred.T)

scores = np.array(k_scores).mean(axis=0)#每个k对应的异常专家的得分

dict_ = {'anomaly_score':scores}

scores = pd.DataFrame(dict_)
scores.index = scores.index + 1
scores.sort_values("anomaly_score", inplace=True,ascending=False)
print(scores)
'''
#预测模型
y = model._predict(x) #若样本点正常，返回1，不正常，返回-1
#显示异常结果
pred = np.array(['正常' if i==1 else '异常' for i in y])
scores_pred = -model._decision_function(x)#decision_function 值越大，越正常；加个负号结论相反。 返回值为一个数组大小跟n_samples一样
dict_ = {'anomaly_score':scores_pred}
scores = pd.DataFrame(dict_)
scores.sort_values("anomaly_score", inplace=True,ascending=False)
print(scores)




dict_ = {'anomaly_score':scores_pred, 'y_pred':y, 'result':pred}
scores = pd.DataFrame(dict_)
print(scores)
'''
