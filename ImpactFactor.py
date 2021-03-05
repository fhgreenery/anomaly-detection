import numpy as np
#读取文件
df = np.loadtxt('data/171111.txt',dtype=int,delimiter='\t')
# x=np.array(data,dtype='float')
#print(df)
#统计每个项目中赞成票数
x = np.where(df>=75,1,-1)
y = np.sum(x==1,axis=0)
print(y)
#统计每个项目去掉最值后的平均值
m0 = (np.sum(df,axis=0)-np.max(df, axis=0)- np.min(df,axis=0))/(len(df)-2)
print(m0)
#除去第i个专家后（i=1,...17）每个项目去掉最值后的平均值

m_list = []
for i in range(len(df)):
    m_row = np.delete(df,i,axis=0)
   # m_list.append(m_row)
    m = (np.sum(m_row,axis=0)-np.max(m_row, axis=0)- np.min(m_row,axis=0))/(len(df)-3)
    m_change = m0 - m
    m_list.append(m_change)

m_factor = np.reshape(np.array(m_list),(df.shape[0],df.shape[1]))
np.savetxt('data/171111_factor.csv',m_factor,delimiter=',')
np.savetxt('data/171111_factor_abs.csv',np.fabs(m_factor),delimiter=',')
#print(m_factor[:,0])

'''
m_final_list = []
#加权，归一化

for i in range(len(y)):
    m_final =(y[i]/len(df)) * m_factor[:,i]
    m_final_list.append(m_final)
m_result = np.reshape(np.array(m_final_list),(df.shape[1],df.shape[0]))

#print(m_result.T)

#保存文件

np.savetxt('data/174444_result.csv',np.fabs(m_result.T),delimiter=',')
'''