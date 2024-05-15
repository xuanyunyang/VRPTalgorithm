import pandas as pd
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt

# 数据结构：解
class Sol():
    def __init__(self):
        self.nodes_seq = None  # 解的编码
        self.obj = None  # 目标函数
        self.routes = None  # 解的解码


# 数据结构：网络节点
class Node():
    def __init__(self):
        self.id = 0  # 节点id
        self.name = ''  # 节点名称，可选
        self.seq_no = 0  # 节点映射id
        self.x_coord = 0  # 节点平面横坐标
        self.y_coord = 0  # 节点平面纵坐标
        self.demand = 0  # 节点需求


# 数据结构：全局参数
class Model():
    def __init__(self):
        self.best_sol = None  # 全局最优解
        self.node_list = []  # 需求节点集合
        self.node_seq_no_list = []  # 需求节点映射id集合
        self.depot = None  # 车场节点
        self.number_of_nodes = 0  # 需求节点数量
        self.opt_type = 0  # 优化目标类型
        self.vehicle_cap = 0  # 车辆最大容量

# 处理表格数据
def readXlsxFile(filepath, model):
    # 建议在xlsx文件中，第一行为表头，其中: x_coord,y_coord,demand是必须项；车辆基地数据放在表头下首行
    node_seq_no = -1  # 车辆基地的seq_no值为-1,剩余需求节点的seq_no 依次编号为 0,1,2,...
    df = pd.read_excel(filepath)
    for i in range(df.shape[0]):
        node = Node()
        node.id = node_seq_no
        node.seq_no = node_seq_no
        node.x_coord = df['x_coord'][i]
        node.y_coord = df['y_coord'][i]
        node.demand = df['demand'][i]
        if df['demand'][i] == 0:
            model.depot = node
        else:
            model.node_list.append(node)
            model.node_seq_no_list.append(node_seq_no)
        try:
            node.name = df['name'][i]
        except:
            pass
        '''
        try:
            node.id = df['id'][i]
        except:
            pass
        '''
        node_seq_no = node_seq_no + 1
    model.number_of_nodes = len(model.node_list)

# 构造初始解
def genInitialSol(node_seq):
    node_seq = copy.deepcopy(node_seq)  # 完全独立的副本,以便在后续的操作中不会对原始的node_seq产生影响
    random.seed(0)
    random.shuffle(node_seq)  # 对node_seq进行随机洗牌。改变node_seq中元素的顺序
    return node_seq

# 采用Split思想对TSP序列进行切割，得到可行车辆路径
def splitRoutes(nodes_seq, model):
    """
    采用简单的分割方法：按顺序依次检查路径的容量约束，在超出车辆容量限制的位置插入车场。
    例如某TSP解为：[1,2,3,4,5,6,7,8,9,10],累计需求为：[10,20,30,40,50,60,70,80,90,10]，车辆容量为：30，则应在3,6,9节点后插入车场，
    即得到：[0,1,2,3,0,4,5,6,0,7,8,9,0,10,0]
    """
    num_vehicle = 0
    vehicle_routes = []
    route = []
    remained_cap = model.vehicle_cap
    for node_no in nodes_seq:
        if remained_cap - model.node_list[node_no].demand >= 0:
            route.append(node_no)
            remained_cap = remained_cap - model.node_list[node_no].demand
        else:
            vehicle_routes.append(route)
            route = [node_no]
            num_vehicle = num_vehicle + 1
            remained_cap = model.vehicle_cap - model.node_list[node_no].demand
    vehicle_routes.append(route)
    return num_vehicle, vehicle_routes

# 计算目标函数
def calObj(nodes_seq, model):
    print(nodes_seq)
    num_vehicle, vehicle_routes = splitRoutes(nodes_seq, model)
    if model.opt_type == 0:
        return num_vehicle, vehicle_routes
    else:
        distance = 0
        for route in vehicle_routes:
            distance += calDistance(route, model)
        return distance, vehicle_routes

# 定义邻域算子
def createActions(n):
    action_list = []
    nswap = n // 2  # 整数除法,去除余数
    # 第一种算子（Swap）：前半段与后半段对应位置一对一交换
    for i in range(nswap):
        action_list.append([1, i, i + nswap])
    # 第二种算子（DSwap）：前半段与后半段对应位置二对二交换 double swap
    for i in range(0, nswap, 2):
        action_list.append([2, i, i + nswap])
    # 第三种算子（Reverse）：指定长度的序列反序
    for i in range(0, n, 4):
        action_list.append([3, i, i + 3])
    return action_list


def run(filepath, T0, Tf, deltaT, v_cap, opt_type):
    """
    :param filepath: xlsx文件路径
    :param T0: 初始温度
    :param Tf: 终止温度
    :param deltaT: 温度下降步长或下降比例
    :param v_cap: 车辆容量
    :param opt_type: 优化类型:0:最小化车辆数,1:最小化行驶距离
    :return:
    """
    model = Model()
    model.vehicle_cap = v_cap
    model.opt_type = opt_type
    readXlsxFile(filepath, model)
    action_list = createActions(model.number_of_nodes)
    history_best_obj = []
    sol = Sol()
    sol.nodes_seq = genInitialSol(model.node_seq_no_list)
    sol.obj, sol.routes = calObj(sol.nodes_seq, model)
    model.best_sol = copy.deepcopy(sol)
    history_best_obj.append(sol.obj)
    Tk = T0  # 初始温度
    nTk = len(action_list)
    while Tk >= Tf:  # 终止温度
        for i in range(nTk):
            new_sol = Sol()
            new_sol.nodes_seq = doACtion(sol.nodes_seq, action_list[i])  # 每个new_sol扰动一次
            new_sol.obj, new_sol.routes = calObj(new_sol.nodes_seq, model)
            detla_f = new_sol.obj - sol.obj
            if detla_f < 0 or math.exp(-detla_f / Tk) > random.random():
                sol = copy.deepcopy(new_sol)
            if sol.obj < model.best_sol.obj:
                model.best_sol = copy.deepcopy(sol)
        if deltaT < 1:
            Tk = Tk * deltaT
        else:
            Tk = Tk - deltaT
        history_best_obj.append(model.best_sol.obj)
        print("当前温度：%s，local obj:%s best obj: %s" % (Tk, sol.obj, model.best_sol.obj))
    # plotObj(history_best_obj)
    # plotRoutes(model)
    outPut(model)
    outPut2(model)