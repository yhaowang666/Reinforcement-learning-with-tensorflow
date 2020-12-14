# -*- codeing = utf-8 -*-
# @Time : 2020/12/9 13:34
# @Author : 王浩
# @File : Sat_IoT_env.py
# @Software : PyCharm

import numpy as np
import time
# import sys
import math


eta = 0.5  # 时延和能耗的比重
K = 5   # 总任务个数
Sat_N = 2    # 定义总卫星个数
Gat_N = 2    # 定义总网关个数
Ter_N = 2    # 定义地面IoT终端的个数
N_k = [8*10**4, 1.2*10**5]   # 定义任务大小区间
w_k = 1   # 每个任务所占的比重
Height = 10**6    # LEO卫星的高度
Cover_radius = 10**6   # LEO卫星的覆盖半径
cycles_average_bit = 1000  # CPU运算能力，单位cycles/bit
Energy_cost = 3**-10   # CPU每循环一圈的能源消耗
X_s = 10*10**6  # 单个卫星的总通信容量
Y_g = 50*10**6  # 单个网关的总通信容量
Z_s = 10**10   # 单个卫星的总计算容量，单位cycles/s
Q_g = 50*10**10  # 单个网关的总计算容量，单位cycles/s
J = 20   # 最大完成回报


class Sat_IoT(object):
    # 实例的初始化
    def __init__(self):
        super(Sat_IoT, self).__init__()
        # self.title('Sat_IoT')
        self._build_Sat_IoT()

    def _build_Sat_IoT(self):
        # 状态空间的定义和初始化
        self.U = np.random.randint(N_k[0], N_k[1], size=K).reshape(K, 1)  # 初始化总任务集
        self.omega = np.zeros((K, Sat_N))   # 初始化与卫星有关的任务集，列为总卫星个数
        # omega_l = np.append(omega_l, [[1,0]], axis = 0)
        self.psi = np.zeros((K, Sat_N))       # 初始化由卫星处理的任务集，列为总卫星个数
        self.phi = np.zeros((K, Gat_N))      # 初始化与网关有关的任务集，列为总网关个数
        self.X = np.zeros((K, Sat_N))       # 初始化卫星正在进行的任务占用的通信资源，列为总卫星个数
        self.X_remain = np.ones((1, Sat_N))*X_s  # 卫星剩余的通信资源
        self.X_allocation = np.zeros((1, Sat_N))
        self.Y = np.zeros((K, Gat_N))       # 初始化网关正在进行的任务占用的通信资源，列为网关总个数
        self.Y_remain = np.ones((1, Gat_N)) * Y_g  # 网关剩余的通信资源
        self.Q = np.zeros((K, Gat_N))      # 初始化网关正在进行的任务占用的计算资源，列为网关总个数
        self.Q_remain = np.ones((1, Gat_N)) * Q_g  # 网关剩余的计算资源
        self.Gat_allocation = np.zeros((1, Gat_N))
        self.Z = np.zeros((K, Sat_N))       # 初始化卫星正在进行的任务占用的计算资源，列为总卫星个数
        self.Z_remain = np.ones((1, Sat_N)) * Z_s  # 卫星剩余的计算资源
        self.Z_allocation = np.zeros((1, Sat_N))
        # 初始化位置矩阵
        # self.PL = np.array([[1.1, 1.2, 1.2, 1.3], [1.2, 1.3, 1.1, 1.2]])*10**6
        self.PL = np.random.randint(Height, math.sqrt(Height**2 + Cover_radius**2),
                                    Sat_N*(Gat_N+Ter_N)).reshape(Sat_N, Gat_N+Ter_N)
        self.state = np.concatenate((self.omega, self.psi, self.phi, self.U, self.X, self.Y, self.Q, self.Z), axis=1)

        # 动作空间的定义和初始化
        # self.A1 = np.arange(2)    # 0代表不安排在该时隙，1代表安排在本时隙
        self.A1 = np.arange(Sat_N+1)  # 0代表不安排在该时隙，其他代表与该任务有关的卫星编号
        self.A2 = np.arange(Gat_N+1)      # 0代表由卫星计算，或者由网关计算
        # self.A4 = np.arange(Gat_N)   #
        self.a = np.zeros((2, K))       # 定义动作的类型，用来储存对每个任务的动作
        self.n_actions = ((Sat_N+1)*(Gat_N+1))**K

    # 将action有十进制数字转变为矩阵的方式表示
    def Action(self, action):
        for i in range(K):
            self.a[0][i] = action % (Sat_N+1)
            action = action // (Sat_N+1)
        for i in range(K):
            self.a[1][i] = action % (Sat_N+1)
            action = action // (Sat_N+1)

    # 判断需要分配的总资源以及是否满足限制条件
    def Source_allocation(self):
        # 判断需要分配的总的卫星通信资源
        for i in range(K):
            if self.a[0][i] != 0:
                self.X_allocation[0][int(self.a[0][i]) - 1] += self.U[i][0]
        # 判断需要分配的总的卫星计算资源
        for i in range(K):
            if int(self.a[0][i]) != 0 & int(self.a[1][i]) == 0:
                self.Z_allocation[0][int(self.a[0][i]) - 1] += self.U[i][0]
        # 判断需要分配的总的网关通信和计算资源
        for i in range(K):
            if int(self.a[0][i]) != 0 & int(self.a[1][i]) != 0:
                self.Gat_allocation[0][int(self.a[0][i]) - 1] += self.U[i][0]
        # 判断该动作是否满足要求
        if int(np.min(self.X_remain - self.X_allocation)) < 0 | \
           int(np.min(self.Z_remain/cycles_average_bit - self.Z_allocation)) < 0 | \
           int(np.min(self.Y_remain - self.Gat_allocation)) < 0 | \
           int(np.min(self.Q_remain/cycles_average_bit - self.X_allocation)) < 0:
            return 1
        else:
            return 0

    # 更新剩余的资源
    def Source_remain(self):
        # 更新剩余卫星通信资源
        self.X_remain -= np.sum(self.X, axis=0)
        # 更新剩余卫星计算资源
        self.Z_remain -= np.sum(self.Z, axis=0)
        # 更新剩余网关通信资源
        self.Y_remain -= np.sum(self.Y, axis=0)
        # 更新剩余卫星计算资源
        self.Q_remain -= np.sum(self.Q, axis=0)

    #  更新X,Y,Z,Q,U,psi,phi，对即将执行的新任务分配资源
    def Source_update(self):
        for i in range(K):
            # 更新X,Y,Z,Q,U
            if self.a[0][i] != 0:
                # 如果任务在该时隙处理，则将总任务集当中对应任务的数据大小放在由卫星有关的任务集
                self.omega[i][int(self.a[0][i]) - 1] = self.U[i][0]
                # 分配卫星上的通信资源
                self.X[i][int(self.a[0][i])- 1] = self.X_remain[0][int(self.a[0][i]) - 1] \
                                              * np.sqrt(self.U[i][0] * eta * w_k) / np.sqrt(
                    self.X_allocation[0][int(self.a[0][i]) - 1] * eta * w_k).sum()
                # 判断由卫星计算还是网关计算
                if self.a[1][i] == 0:
                    # 由卫星计算的任务集
                    self.psi[i][int(self.a[0][i]) - 1] = self.U[i][0]
                    # 分配卫星上的计算资源
                    self.Z[i][int(self.a[0][i]) - 1] = self.Z_remain[0][int(self.a[0][i]) - 1] \
                        * np.sqrt(self.U[i][0] * eta * w_k) / \
                        np.sqrt(self.Z_allocation[0][int(self.a[0][i]) - 1] * eta * w_k).sum()
                else:
                    # 与网关有关的任务集
                    self.phi[i][int(self.a[1][i]) - 1] = self.U[i][0]
                    # 分配网关通信资源
                    self.Y[i][int(self.a[1][i]) - 1] = self.Y_remain[0][int(self.a[1][i]) - 1] \
                        * np.sqrt(self.U[i][0] * eta * w_k) / \
                        np.sqrt(self.Gat_allocation[0][int(self.a[1][i]) - 1] * eta * w_k).sum()
                    # 分配网关计算资源
                    self.Q[i][self.a[1][i] - 1] = self.Q_remain[0][int(self.a[1][i]) - 1] \
                        * np.sqrt(self.U[i][0] * eta * w_k) / \
                        np.sqrt(self.Gat_allocation[0][int(self.a[1][i]) - 1] * eta * w_k).sum()
                # 并将U中该位置的任务清零
                self.U[i][0] = 0

    def step(self, action):
        time.sleep(0.3)
        self.Action(action)
        # 更新剩余资源
        self.Source_remain()

        # 更新X_allocation，Z_allocation，Gat_allocation
        if_error = self.Source_allocation()
        if if_error == 1:
            print("动作选择无效，状态不更新")
            return

        # 更新X,Y,Z,Q,U,psi,phi
        self.Source_update()
        self.state = np.concatenate((self.omega, self.psi, self.phi, self.U, self.X, self.Y, self.Q, self.Z), axis=1)

    def show_system(self):
        print(self.state, self.a)











