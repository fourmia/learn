# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:45:34 2019
这是一个cimiss数据处理模块，通过datainterface中的cimissdata
方法获取的数据可以直接被模块中的函数进行处理，enjoy......
@author: GJW
"""
import pandas as pd
import datetime as dt
import pickle
from ..DataInterface import Datainterface
from scipy.interpolate import griddata

def readpickle(path):
    '''
    func: 完成pickle数据解析
    inputs:
        path: 待解析的pickle所在的完整路径
    return:
        result: 解析后的数据结果
    '''
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result

# 此函数可用作数据处理模块完成
def onehour2threehour(data):
    """
    此函数将逐15分钟实况数据或逐一小时实况数据处理成逐三小时的实况数据，便于与模式数据统一起来
    注：data格式应为数据框，通常用来处理Datainterface接口中cimissdata方法获取到的数据
    ：params:
        data:传入获取的初始实况数据
    ：return:
        逐三小时形式的数据
    """
    data.Time = pd.to_datetime(data.Time)
    data.set_index('Time', inplace=True)
    icedata = data.groupby('Station_Id_C').resample('3H',
                                base=8).mean()
    return icedata

def cleandata(data,exdata,*args):
    """
    此函数用于清洗数据去除存在Nan值和异常值的行，注：传入的数据应为数据框格式，
    通常用来处理Datainterface接口中cimissdata方法获取到的数据。
    ：params:
        data:传入需要处理的数据
        exdata:异常值
        args:需要剔除异常值的列名,多个以列表形式传入,须确保函数参数存在
    ：return:
        此处返回数据框格式数据
    """
    for arg in args:
        data = data[data[arg]<exdata]
    return data.dropna()
# 插值算法，点太少会出现问题
def insert2road(dataframe, roaddf, method='cubic', fill_value=0):
    """
    此函数完成了站点到站点的插值，注意：要是原始站点过少，待插值站点过多会出现
    大量nan值,nan值默认以0填充
    ：params:
        dataframe:原始数据，数据框格式，包含Lat、Lon、data列
        roaddf：待插值数据，数据框格式， 包含lat、lon列
        method: 默认cubic，可选linear，nearest可能会报错暂未做异常处理
    ：return
        待插值数据相应经纬度的插值结果，ndarray
    """
    points = dataframe[['Lon','Lat']].to_numpy()
    roadpointvalue = griddata(points, dataframe.Road_ICE_Depth,
                             (roaddf.lon.values.reshape(-1,1), roaddf.lat.values.reshape(-1,1)), method=method, fill_value=fill_value)
    return roadpointvalue



if __name__ == "__main__":
    interfaceId = 'getSurfEleInRegionByTimeRange'
    elements = "Station_Id_C,Lat,Lon,Road_ICE_Depth,Year,Mon,Day,Hour"
    lastime = dt.datetime.now().replace(minute=0, second=0) - dt.timedelta(hours=8)
    fristime = lastime - dt.timedelta(hours=3)
    temp = ('[',fristime.strftime('%Y%m%d%H%M%S'),',',lastime.strftime('%Y%m%d%H%M%S'),')')
    Time = ''.join(temp)
    params = {'dataCode':"SURF_CHN_TRAFW_MUL",
      'elements':elements,
      'timeRange':Time,
      'adminCodes':"630000"
      }
    initnaldata = Datainterface.cimissdata(interfaceId, elements, **params)
    o2tdata = onehour2threehour(initnaldata)
    nonandata = cleandata(o2tdata, 999, 'Road_ICE_Depth')
    #####################################################################
    roadpath = r'E:\LZD\青海项目代码\qhroadic\eightroadmessage.pickle'
    roaddf = readpickle(roadpath)
    #####################################################################
    roadvalue = insert2road(nonandata, roaddf)       # 得到插值后道路点的结冰数据，坐标则是有roaddf的lat、lon提供
    roadvalue = pd.DataFrame(roadvalue)
    # roadvalue.to_csv(r'C:\Users\GJW\Desktop\test.csv', index=None, mode=a)
    cmissdata = pd.read_csv(r'C:\Users\GJW\Desktop\test.csv')
    cmissdata[len(cmissdata.columns)] = roadvalue
    cmissdata.to_csv(r'C:\Users\GJW\Desktop\test.csv',mode='a',index=None)
    

