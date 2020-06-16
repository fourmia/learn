import numpy as np
import pandas as pd


def svrformat(data):
    # svr输入
    tmp = np.array([i.flatten() for i in data])
    return tmp

def fcnformat(data, path=None):
    """
    将数据转换格式，符合全卷积神经网络输入
    :param args: 提供的数据列表
    :param path: 提供的高程数据文件名称，不需要高程信息则为None
    :param num:  神经网络输入要素数量
    :return: 处理后的数据网格
    """
    if path is None:
        print("The dataset donot exist DEM message!")
        tmp = data
    else:
        dem = pd.read_csv(path)
        dem = dem.iloc[:, 1:].to_numpy()
        tmp = np.array([it, dem] for it in data)
    return tmp





