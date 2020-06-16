import datetime
import re
import numpy as np


def znwgtime():
    """16点前获取今天08时， 16点后获取今天20时
    获取时间一般用于智能网格预报"""
    now_time = datetime.datetime.now()
    if now_time.strftime('%H') >= '04' and now_time.strftime('%H') <'16':
        report = now_time.replace(hour=8, minute=0, second=0)
    if now_time.strftime('%H') >= '16' or now_time.strftime('%H') < '04':
        report = now_time.replace(hour=20, minute=0, second=0)
    return report


def arrange(*args):
    """
    The total is organize grib data from Datainterface.Grib
    Tips: the function data is same type data
    :param args: data---> dict type{paraName_time1: array1,
                                    paraName_time2: array2...}   ----->array1、2... is ndarray
                  lat---> list type [array1_lat, array2_lat....]------> array1_lat... is ndarray
                  lon---> list type [array1_lon, array2_lon....]------> array1_lon... is ndarray
                  size --> Resolution :list type [float]                    ------> List
    :return:
                 data---> list[array1, array2...]
                 lat--->  ndarray
                 lon--->  ndarray
                 size ---> floar
    """
    data = np.array([v for _, v in args[0][0].items()])  #-->提取解析数据
    lat, lon, size = args[0][1][0], args[0][2][0], args[0][3][0]    #-->同一存储提取出一个信息
    return data, lat, lon, size

def regex(pattern, strings):
    if isinstance(strings, list):
        strings = ','.join(strings)
    if not isinstance(strings, str):
        raise TypeError("parament strings is not str type or List type!")
    re.compile(pattern)
    return re.findall(pattern,strings)
