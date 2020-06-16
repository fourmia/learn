# 此模块保存Product package 共享变量
import numpy as np
import datetime as dt
from Dataprocess import Znwg
import pandas as pd


# 一维数组，用来保存NC文件
lat = np.arange(31, 40.001, 0.01)
lon = np.arange(89, 103.001, 0.01)

# 二维网格，用以进行格点插值
longrid, latgrid = np.meshgrid(lon, lat)

# 国家级降雨数据经纬度范围裁剪
# 国家级降雨数据经纬度范围裁剪, 按照zhwg文件范围裁剪
latt = np.linspace(31.4, 39.4, 161)
lonn = np.linspace(89.25, 102.95, 274)

# 写入NC文件必要信息
now = dt.datetime.now()
filetime = Znwg.znwgtime().strftime('%Y%m%d%H%M')
fh = range(3, 169, 3)
fnames = ['_%03d' % i for i in fh]


# 配置文件路径及道路文件路径
windpath = r'/home/cqkj/QHTraffic/Product/Product/Source/Road_wind.csv'
roadpath = r'/home/cqkj/QHTraffic/Product/Product/Source/QHroad_update.csv'
trafficpath = r'/home/cqkj/QHTraffic/Product/Product/config/Traffic.xml'
forestpath = r'/home/cqkj/QHTraffic/Product/Product/config/forest.xml'
