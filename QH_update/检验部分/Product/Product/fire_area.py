from Product import glovar

# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:58:18 2020
# 此函数用来计算过火面积
@author: GJW
"""
import os
import time
import requests
import csv
import datetime as dt
import pandas as pd
from netrc import netrc
import pandas as pd
import numpy as np
import math
from retrying import retry
from scipy.spatial import ConvexHull

@retry(stop_max_attempt_number=10)
def spider():
    # 同步fire_activate文件
    # 待下载数据的下载链接
    now = dt.datetime.now().strftime('%Y%m%d')
    f = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Russia_Asia_24h.csv"
    # 数据的存在位置
    saveName = ''.join([r'C:\Users\GJW\Desktop\MODIS\\', r'J1_VIIRS_C2_Russia_Asia_', now, r'.csv'])
    # 数据的保存名称：和数据下载名称一致
    print(saveName)
    # 授权信息的存放路径
    netrcDir = os.path.expanduser(".netrc")
    # 带授权的网址
    urs = 'Earthdata Login'  # Earthdata URL to call for authentication
    print(urs)
    print(f.strip())
    # print(netrc(netrcDir).authenticators(urs)[0], netrc(netrcDir).authenticators(urs)[2])
    # Create and submit request and download file
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, \
                   like Gecko) Chrome/47.0.2526.106 Safari/537.36"}
    with requests.get(f.strip(), headers=headers) as response:
        if response.status_code != 200:
            print("file not downloaded. Verify that your url are correct")
        else:
            start = time.time()  # 开始时间
            size = 0  # data of download
            content_size = int(response.headers['Content-Length'])  # 总大小
            print('file size:' + str(content_size) + ' bytes')
            print('[文件大小]：%0.2f MB' % (content_size / 1024 / 1024))

            response.raw.decode_content = True
            content = response.text
            cr = csv.reader(content.splitlines(), delimiter=',')
            my_list = list(cr)
            df = pd.DataFrame(my_list[1:], columns=my_list[0])
            df.to_csv(saveName)
            end = time.time()  # 结束时间
            print('\n' + "全部下载完成！用时%0.2f秒" % (end - start))
            # dataset[(dataset.longitude<103.1) & (dataset.longitude>89.3) &
            #  (dataset.latitude>31.4) & (dataset.latitude<39.4)]


def Node(QHpoints, distance=0.5):
    # 此程序将节点分类，获取青海省当前节点指定区域内的其它节点
    # 保存为List
    res = []
    for i in range(len(QHpoints) - 1):
        lon, lat = QHpoints[i]
        tmp = [QHpoints[i].tolist()]
        for lonn, latt in QHpoints[i + 1:]:
            if (lonn > (lon - distance)) & (lonn < (lon + distance)) & (lat > (lat - distance)) & (
                lat < (lat + distance)):
                tmp.append([lonn, latt])
        res.append(tmp)
    return res


def clcfire(df, list2, list3):
    # 1个节点
    scan_track = df[['scan', 'track']]
    res = np.array(scan_track)[:, 0] * np.array(scan_track)[:, 1]
    # 2个节点
    array2 = np.array(list2)
    testarray = np.sqrt(abs(np.diff(array2[0], axis=0)) * pow(10, 5))
    res2 = math.hypot(testarray[0][0], testarray[0][1])
    # 3个节点
    for sublist in list3:
        hull = ConvexHull(sublist)
        print(hull.area)
    return res, res2, hull.area


path = r'C:\Users\GJW\Desktop\MODIS\J1_VIIRS_C2_Russia_Asia_24h.csv'
dataset = pd.read_csv(path).iloc[:, 1:]
Qinghai = dataset[(dataset.latitude >= 31.4) & (dataset.latitude <= 39.4) & (dataset.longitude >= 89.3) &
                  (dataset.longitude <= 103.1)]
QHpoints = Qinghai[['latitude', 'longitude']].values
tests = Node(QHpoints)
list1, list2, list3 = [], [], []  # 存放不同节点个数的数组
for test in tests:
    print(len(test))
    if len(test) < 2:
        list1.append(test)
    elif len(test) >= 3:
        list3.append(test)
    else:
        list2.append(test)


def main():
    pass


if __name__ == '__main__':
    main()