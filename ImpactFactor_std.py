import numpy as np

# 取消numpy默认使用科学计数法
np.set_printoptions(suppress=True)
# 读取文件
df = np.loadtxt('data/174444.txt', dtype=int, delimiter='\t')
# print(df)
# 统计每个项目中赞成票数
x = np.where(df >= 75, 1, -1)
y = np.sum(x == 1, axis=0)
# print(y)
# 统计每个项目去掉最值后的平均值
m0 = (np.sum(df, axis=0) - np.max(df, axis=0) - np.min(df, axis=0)) / (len(df)-2)
# print(m0)
# 除去第i个专家后（i=1,...17）每个项目去掉最值后的平均值
m_list = []
m2_list = []
std_list = []
for i in range(len(df)):
    m_row = np.delete(df, i, axis=0)
    m = (np.sum(m_row, axis=0) - np.max(m_row, axis=0) - np.min(m_row, axis=0)) / (len(df) - 3)
    m_change = np.abs(m0 - m)
    m_list.append(m_change)
    # 求平方矩阵
    m2 = np.power(m_change, 2)
    m2_list.append(m2)

m2_array = np.reshape(np.array(m2_list), (df.shape[0], df.shape[1]))
# m_array = np.reshape(np.array(mean_list), (17, 18))
m_factor = np.reshape(np.array(m_list), (df.shape[0], df.shape[1]))

for i in range(len(df)):
    std = np.power((np.sum(m2_array, axis=0) - m2_array[i, :]) / (len(df)-2), 1 / 2)
    std_list.append(std)
std_array = np.reshape(np.array(std_list), (df.shape[0], df.shape[1]))
score = m_factor / std_array

# 票数加权
weight_list = []
for i in range(len(y)):
    weight = (y[i] / len(df)) * score[:, i]
    weight_list.append(weight)
result = np.reshape(np.array(weight_list).T, (df.shape[0], df.shape[1]))
np.savetxt('data/174444_score_std.csv',result,delimiter=',')
