import numpy as np
import pandas as pd
#读取文件
df = np.loadtxt('data/194444.txt',dtype=int,delimiter='\t') #193333
rows,cols = df.shape
#求每一列期望
Ex = df.mean(0)

#每一元素与该列期望差的绝对值
def abs_data(df, Ex):
    abs_list = []
    for i in range(rows):
        for j in range(cols):
            #绝对值
            abs_data = np.abs(df[i][j]-Ex[j]) 
            abs_list.append(abs_data)
    #转化成矩阵
    abs_array = np.reshape(np.array(abs_list),(len(df),len(Ex)))

    return abs_array

#计算异常程度
def abnormal_score(abs_array,LM):
    abnormal_list = []
    for i in range(rows):
        for j in range(cols):
            t = abs_array[i][j]/LM
            abnormal_list.append(t)
    abnormal_array = np.reshape(np.array(abnormal_list),(rows,cols))
    return abnormal_array

abs_array = abs_data(df,Ex)
# Ee
Ee= (np.power(np.pi/2,1/2)/rows) * abs_array.sum(axis=0)

#print('Ee',Ee)
#s2
pow_array = np.power(abs_array,2)
s2 = pow_array.sum(axis=0)/(rows-1)
#print('s2',s2)
# He
He = np.power(np.abs(s2 - np.power(Ee,2)),1/2)
#print('He',He)
#计算L
L = 3*Ee + 9*He
#print('L',L)
LM = L.mean()
#print('LM:',LM)
abnormal = abnormal_score(abs_array,LM)
scores = abnormal.mean(axis=1)
dict_ = {'anomaly_score':scores}
scores = pd.DataFrame(dict_)

scores.index = scores.index + 1
scores.sort_values("anomaly_score", inplace=True,ascending=False)
#print(scores)
scores.to_excel("data/194444_cloud.xls")
'''
temp2 = abnormal.reshape(1,-1) #array一维数组表示一行n列
temp3 = np.argsort(-temp2) # 0-rows*cols-1 的索引 -X 降序
result = pd.DataFrame(abnormal)
result.index = result.index + 1
result.to_excel("data/193333_cloud.xls")
for i in range(7) : #单行索引转化成矩阵索引
    temp4 = temp3[0][i]+1
    i_index =temp4//25 + 1
    j_index = temp4 % 25
    print((i_index,j_index))
'''

#怎么判定异常专家
'''
1.计算每个专家异常得分的均值
2.topK个异常评审数据中该专家所占的个数等等
'''

