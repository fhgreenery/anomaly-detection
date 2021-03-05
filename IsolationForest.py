#coding=utf-8
# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest  


'''
API简要说明：
sklearn.ensemble.IsolationForest(n_estimators=100, max_samples='auto', contamination='legacy', max_features=1.0, 
                                 bootstrap=False, n_jobs=None, behaviour=’old’, random_state=None, verbose=0)
n_estimators：iTree的个数；
max_samples：构建单颗iTree的样本数；
contamination：异常值的比例；
max_features：构建单颗iTree的特征数；
bootstrap：布尔型参数，默认取False，表示构建iTree时有放回地进行抽样；
'''
#,usecols=(6,9,10) n_estimators=30,usecols=(0,1,2,3,4)
outliers_fraction = 0.18
n_samples = 17
rng = np.random.RandomState(36)
X_train = np.loadtxt('data/193333.txt',dtype=int,delimiter='\t')


clf = IsolationForest(n_estimators=100,max_samples=n_samples,contamination=outliers_fraction,random_state=None,n_jobs=-1, behaviour="new")
# predict / fit_predict方法返回每个样本是否为正常值，若返回1表示正常值，返回-1表示异常值
y_pred_train = clf.fit_predict(X_train)#-1 or 1
pred = np.array(['正常' if i==1 else '异常' for i in y_pred_train])

# 分数越小于0，越有可能是异常值
scores_pred = clf.decision_function(X_train) 
dict_ = {'anomaly_score':scores_pred, 'y_pred':y_pred_train, 'result':pred}
scores = pd.DataFrame(dict_)
print(scores)


