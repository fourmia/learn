# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:19:58 2020
此程序用来计算过火面积
@author: GJW
"""
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial import distance
import networkx as nx
import requests
import json
import os
import datetime as dt
import math
import time
import csv
from retrying import retry


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
    #netrcDir = os.path.expanduser(".netrc")
    # 带授权的网址
    #urs = 'Earthdata Login'  # Earthdata URL to call for authentication
    #print(urs)
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



def pointdis(latlon):
    """计算两个投影坐标之间的欧式距离, 返回对称距离矩阵，单位(m)"""
    # 参数：proj 多个横纵坐标对，形如[382049.91503404, -164938.75931201]
    return distance.cdist(latlon, latlon, 'euclidean') * pow(10, 5)


def graph(dataset, latlons, distance=500):
    """根据输入的距离矩阵和规定的联通范围来设定edge,最终返回列表格式的所有子图"""
    # dataset 对称距离矩阵；latlons
    # 计算出经纬度的分类结果
    node = np.where((dataset < 500) & (dataset > 0))
    G = nx.Graph()
    for lat, lon, scan, track in zip(latlons[:, 0], latlons[:, 1], latlons[:, 2], latlons[:, 3]): G.add_node((lat, lon, scan, track))
    for x, y in zip(*node): G.add_edge(tuple(latlons[x]), tuple(latlons[y]))
    # [G.add_node((lat, lon)) for lat,lon in zip(latlons[:,0], latlons[:,1])]
    # [G.add_edge(tuple(latlons[x]), tuple(latlons[y])) for x, y in zip(*node)]
    return list(nx.connected_components(G))


def clculate(subgraph, dataset, X=None, tag=1):
    # 根据前几天的火点状态及今天的火点信息得到过火面积
    # 参数： latlonlist,为今天前的火点连通图， X为昨天前的火点结果信息
    print('tag:{}'.format(tag))
    now = dt.datetime.now().strftime('%Y%m%d')
    # now = '20200520'
    Z = {}
    i = 0
    # 替换掉每个area计算， 考虑使用向量类
    for sub in subgraph:
        print(sub)
        if len(sub) == 1:
            # 判断其是否在dataset里面，在则提取出scan、track否则默认375，最后去掉
            # print(list(sub)[0][0], list(sub)[0][1])
            Z[now + '_' + str(i)] = {'latlon': list(sub), 'area': list(sub)[0][2] * list(sub)[0][3] * pow(10, 6)}
        elif len(sub) == 2:
            # 通过向量法来计算两个火点间的过火面积
            # print(list(sub)[1][0])
            s0 = math.hypot(list(sub)[0][2]*pow(10, 3), list(sub)[0][3]*pow(10, 3))    # 第一个斜边
            s1 = math.hypot(list(sub)[1][2]*pow(10, 3), list(sub)[1][3]* pow(10, 3))    # 第二个斜边
            h = math.hypot((list(sub)[1][0] - list(sub)[0][0]) * pow(10, 5), (list(sub)[1][1] - list(sub)[0][1]) * pow(10, 5))
            print(s0, s1, h)
            area = (s0+s1) * h / 2
            Z[now + '_' + str(i)] = {'latlon': list(sub), 'area': area}
        else:
            Z[now + '_' + str(i)] = {'latlon': list(sub), 'area': ConvexHull(np.array(list(sub))[:, :2]).area * pow(10, 6)}
        i += 1

    if tag == 2:
        j = 0
        for k1, v1 in Z.items():
            for k2, v2 in X.items():
                if (set(v1['latlon']) == set([tuple(e) for e in v2['latlon']])):
                    Z[j] = Z.pop(k1)
                    j += 1
                elif set([tuple(e) for e in v2['latlon']]) < set(v1['latlon']):
                    Z[k2] = Z.pop(k1)

    for key in list(Z):
        if int(key) < 2020:
            del Z[key]

    return Z


def main():
    # 程序入口
    spider()
    path = r'C:\Users\GJW\Desktop\MODIS\\'
    if os.path.exists(path):
        fname = sorted(os.listdir(path))[-1]
        abspath = path + fname
    else:
        raise FileNotFoundError('The path is not exist, please check it!')
    print(abspath)
    dataset = pd.read_csv(abspath, index_col=0)  # 读取出同步文件信息
    dataset = dataset[(dataset.latitude >= 31.4) & (dataset.latitude <= 39.4) & (dataset.longitude >= 89.3) & (
    dataset.longitude <= 103.1)]
    dataset.acq_time = ['%04d' % var for var in dataset.acq_time.values]
    dataset.acq_date = ','.join(dataset.acq_date.values).replace('-', '').split(',')
    dataset['times'] = [dt + tm for dt, tm in zip(dataset.acq_date.values, dataset.acq_time.values)]
    latlons = dataset[['latitude', 'longitude', 'scan', 'track']].values
    #
    ########################################################
    yesterday = r'C:\Users\GJW\Desktop\青海项目整理最新版\Product\Firedata\\'                        # 须确保该目录下仅有json文件
    if not os.path.exists(yesterday):
        pre = None
    else:
        ytime = dt.datetime.now() - dt.timedelta(days=1)
        yname = sorted(os.listdir(yesterday))[-1]
        ypath = yesterday + yname
        if ypath.find(ytime.strftime('%Y%m%d'))>-1:
            with open(ypath) as f:
                pre = json.load(f)
        else:
            pre = None
    ########################################################
    if pre:
        print('yesterday exist Fire!')
        tag = 2
        prelist = [ele for k, v in pre.items() for ele in v['latlon']]
        totalist = np.concatenate([prelist, latlons])
    else:
        tag = 1
        totalist = latlons
        pre = None
    ########################################################
    dis = pointdis(totalist)
    subgraph = graph(dis, totalist, distance=500)
    result = clculate(subgraph, dataset, pre, tag)
    print(result.values())
    ##########################################################
    # now = '20200520'
    now = dt.datetime.now().strftime('%Y%m%d')
    savename = now + r'.json'
    with open(savename, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()
