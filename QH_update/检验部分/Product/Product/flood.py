import os
import pickle
import numpy as np
import pandas as pd
import datetime
from Dataprocess import Writefile, interp, Znwg
from DataInterface import Datainterface

import glovar
from Model import FloodModel
# 此模块为青海积水子模块


def predepth():
    # 前一时刻积水深度,此处需在服务器端测试优化
    dr = np.zeros(shape=(901, 1401))   # 目前默认前一时刻积水深度为0
    now = datetime.datetime.now()
    znwgtm = Znwg.znwgtime()
    *_, ftp = Writefile.readxml(glovar.trafficpath, 1)
    ftp = ftp.split(',')
    grib = Datainterface.GribData()
    remote_url = os.path.join(r'\\ANALYSIS\\CMPA\\0P05', now.strftime('%Y'), now.strftime('%Y%m%d'))
    localdir = r'/home/cqkj/QHTraffic/Product/Product/mirror/rainlive'
    grib.mirror('FRT_CHN_0P05_3HOR-PRE', remote_url, localdir, ftp)
    rname = sorted(os.listdir(localdir))[-1]
    rpath = os.path.join(localdir, rname)
    data, lat, lon, _ = Znwg.arrange((grib.readGrib(rpath)))
    data = interp.interpolateGridData(data, lat, lon, glovar.lat, glovar.lon)
    dataset = data[np.newaxis, ]                # 符合形状要求
    print(dataset.shape)
    res = FloodModel.cal2(dataset, dr)
    return res[0]


def rainData():
    # 同步降雨智能网格文件并解析
    now = datetime.datetime.now()
    *_, elements, ftp = Writefile.readxml(glovar.trafficpath, 1)
    #*_, elements, ftp = Writefile.readxml(r'/home/cqkj/LZD/Product/Product/config/Traffic.xml', 5)
    element = elements.split(',')
    ftp = ftp.split(',')
    grib = Datainterface.GribData()
    remote_url = os.path.join(r'\\SPCC\\BEXN', now.strftime('%Y'), now.strftime('%Y%m%d'))
    grib.mirror(element[0], remote_url, element[1], ftp, element[2])
    rname = sorted(os.listdir(element[1]))[-1]
    rpath = element[1] + rname
    dataset, lat, lon, _ = Znwg.arrange((grib.readGrib(rpath)))    # result包含data,lat,lon,size
    return [interp.interpolateGridData(data, lat, lon, glovar.lat, glovar.lon) for data in dataset[:56]]


def stagnant(savepath, predepth, zdata):
    # 得到道路积水深度数据
    data = zdata
    rdata = np.nan_to_num(data[:56])
    mes = pd.read_csv(glovar.roadpath, index_col=0)
    stationlon, stationlat = mes['Lon'].values, mes['Lat'].values  # 提取出道路坐标
    datas = FloodModel.cal2(rdata, predepth)

    #data = FloodModel.cal2(data, dr)
    # 此处需要加快速度，插值到道路点的速度过慢
    roadpoint = [interp.grid_interp_to_station([glovar.longrid, glovar.latgrid, data],
                                          stationlon, stationlat) for data in datas]
    result = np.nan_to_num(roadpoint)
    result[result < 0] = 0
    flooding_index = np.piecewise(result, [result <=0.4, (result > 0.4) & (result <= 1.0),
                                   (result > 1.0) & (result <= 1.5),(result > 1.5) & (result <= 2.0), result > 2.0], [1, 2, 3, 4, 5])

    Writefile.write_to_csv(savepath, flooding_index, 'floodindex', glovar.fnames, glovar.filetime)
    return roadpoint


def main():
    savepath = r'/data/QHindustry/QHtraffic/index/flood/'
    dr = predepth()
    znwgdata = rainData()
    stagnant(savepath, dr, znwgdata)


if __name__ == "__main__":
    main()
