# -*- coding:utf-8 -*-
'''
使用模拟退火法求100个点的最短路径
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import arccos
#从文件中加载数据
a = np.loadtxt('Pdata17_1.txt')
#a为一二维列表，取每行中奇数位数据，降维成一维列表后存放x中
x = a[:, ::2].flatten()
#a为一二维列表，取每行中偶数位数，降维成一维列表后据存放y中
y = a[:, 1::2].flatten()
#将起始位的经纬度存放在变量d中
d1 = np.array([[70, 40]])
#将经纬度组成对，存放于二维列表xy中
xy = np.c_[x, y]
#将起始位的经纬度添加到xy列表的收尾处
xy = np.r_[d1, xy, d1]
#求xy列表中的行数
N = xy.shape[0]
# 将xy列表中的经纬度转换为弧度，存于列表t中
t = np.radians(xy)
#计算表中各个点在地球上的距离，存于列表中
d = np.array([[6370 * arccos(np.cos(t[i, 0] - t[j, 0]) * np.cos(t[i, 1]) * np.cos(t[j, 1]) + \
                                np.sin(t[i, 1]) * np.sin(t[j, 1])) for i in range(N)] for j in range(N)]).real
#path记录遍历各点顺序的列表
path = np.arange(N)
#L记录遍历各点后距离的最小值
L = np.inf
#使用蒙特卡洛方法为模拟退火方法取一个初始值
for j in range(10000):
    path0 = np.arange(1, N - 1)
    np.random.shuffle(path0)
    path0 = np.r_[0, path0, N - 1]
    L0 = d[0, path0[1]]

    for i in range(1, N - 1):
        L0 += d[path0[i], path0[i + 1]]
    if L0 < L:
        path = path0
        L = L0
#e为温度下降的终点
e = 0.1 ** 30
#M为循环的最大次数
M = 100000
#温度下降率
at = 0.999
#温度的初始值
T = 1
# print(L)
max_k = 0
for k in range(M):
    #在1~100中随机选2个数据作为节点的下标
    c = np.random.randint(1, 101, 2)
    # 2个数据排序
    c.sort()
    # 小值放在c1中，大值放在c2中
    c1 = c[0]
    c2 = c[1]
    # 比较下标为c1和c2两个点交换顺序后的距离变化
    # 如果小于0，说明交换下标为c1和c2两个点的位置缩短了整体的距离
    # 调整点的排序，并存储于path中，计算整体的距离，存放于L中
    df = d[path[c1 - 1], path[c2]] + d[path[c1], path[c2 + 1]] - \
         d[path[c1 - 1], path[c1]] - d[path[c2], path[c2 + 1]]

    if df < 0:
        path = np.r_[path[0], path[1:c1], path[c2:c1 - 1:-1], path[c2 + 1:102]]
        L += df
    # 如果小于0，说明交大换下标为c1和c2两个点的位置增加了整体的距离
    # 按照一定的概率，选择是否采用这条路径
    # 如果采用，点的排序存储于path中，计算整体的距离，存放于L中
    else:
        if np.exp(-df / T) >= np.random.rand(1):
            path = np.r_[path[0], path[1:c1], path[c2:c1 - 1:-1], path[c2 + 1:102]]
            L += df
    # 每循环一次，交换一次点的位置，计算点连成线后的距离，温度下降一次
    # 本质上讲温度控制循环的次数
    T *= at
    max_k = k
    # 当温度小于终点温度值时，停止循环
    if T < e:
        break

print(max_k)
print(path, '\n', L)
xx = xy[path, 0]
yy = xy[path, 1]
plt.rcParams['font.size'] = 16
plt.plot(xx, yy, '-*')
plt.show()
