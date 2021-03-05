
import numpy as np
t = [63.66, 9, 925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169, 3.106, 3.055, 3.012, 2.977, 2.947]
# 读取文件
df = np.loadtxt('data/171111.txt', dtype=int, delimiter='\t')
# print(df)
# 排序
df_sort = np.sort(df, axis=0)
data = df_sort[:, 0]
i = 0;
while (i < 18):
    # 计算中位数
    # data = df_sort[:,i]
    #print(data)
    mid = np.median(data)
    #print(mid)
    # 计算最大偏差
    bias_index = 0 if np.abs(data[0] - mid) >= np.abs(data[-1] - mid) else -1
    bias = np.abs(data[0] - mid) if np.abs(data[0] - mid) >= np.abs(data[-1] - mid) else np.abs(data[-1] - mid)
    #print(bias)
    df_del = np.delete(data, bias_index, 0)
    #print(df_del)
    sum = 0
    for j in range(len(df_del)):
        temp = np.power(df_del[j] - mid, 2)
        sum += temp
    std = np.power(sum / (len(df_del) - 1), 1 / 2)
    # print(std)
    # 计算异常得分
    score = bias / std
    # print(score)
    if i >= 17:
        break;
    # 判断是否异常
    if score >= t[len(df_del) - 1]:

        print('项目 {}：{} 为异常分数, 异常值为{}'.format(i + 1, data[bias_index], score))
        data = df_del
    else:
        print('项目 {}：{} 非异常分数, 异常值为{}'.format(i + 1, data[bias_index], score))
        i = i + 1
        data = df_sort[:, i]