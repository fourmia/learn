# Uodate write_to_nc, write_to_csv
import datetime
import xml.etree.cElementTree as ET
import pymysql
import pandas as pd
import xarray as xr
import numpy as np
import glovar
from Dataprocess import interp, ecmwf, Writefile, Znwg
from DataInterface import Datainterface


def write_to_nc(path, data, name, fnames, filetime, de):
    # dt 为所需提取点数据
    cirnum = data.shape[0]
    print(cirnum)
    for i in range(cirnum):
        print(data[i][np.newaxis,:].shape)
        #print(dt.datetime.strptime(filetime,'%Y%m%d%H%M'))
        # 添加预报入库部分代码
        if i < 4:
            # 对应到待检验站点上，有的话用in，没有则插值, name为df列名
            tmp = np.nan_to_num(interp.interpolateGridData(data[i], glovar.lat, glovar.lon, de.Lat, de.Lon, isGrid=None))
            time = (datetime.datetime.now() + datetime.timedelta(hours=i * 3)).strftime('%Y%m%d%H')     # 此处应换为智能网格time
            alltime = np.repeat(time, len(tmp))
            df = pd.DataFrame(tmp[:, np.newaxis], columns=[name])
            Time = pd.DataFrame(alltime[:, np.newaxis], columns=['Time'])
            de.reset_index(inplace=True)
            dataframe = pd.concat([de, df, Time], axis=1)
            # 入库
        ds = xr.Dataset({name: (['time', 'lat', 'lon'], data[i][np.newaxis,:])}, coords={'time':[de.datetime.strptime(filetime,'%Y%m%d%H%M')],'lat': lat, 'lon': lon})
        f = ''.join([path, filetime, '.', fnames[i], '.', name, '.nc'])
        ds.to_netcdf(f, format='NETCDF3_CLASSIC')


def write_to_csv(path, data, name, fnames, filetime, dt):
    # 此处需添加road文件, dt为所需提取点数据
    cirnum = len(data)
    # 读取road文件合并
    road = pd.read_csv(glovar.roadpath, index_col=0)
    for i in range(cirnum):
        # allpath = path + name + '_' + filetime + fnames[i] + '.csv'
        allpath = path + filetime + '.' + fnames[i] + '.' + name + '.csv'
        dataframes = pd.concat([road, pd.DataFrame(data[i], columns=name, dtype=np.int8)])
        if i < 4:
            # 对应到待检验站点上，dt为待提取的道路点
            tmp = np.array([dataframe[(dataframe.Lat == dtlat) & (dataframe.Lon == dtlon)].name.values for dtlat, dtlon in zip(dt.Lat.values, dt.Lon.values)])
            time = (datetime.datetime.now() + datetime.timedelta(hours=i * 3)).strftime('%Y%m%d%H')     # 此处应换为智能网格time
            alltime = np.repeat(time, len(tmp))
            df = pd.DataFrame(tmp[:, np.newaxis], columns=[name])
            Time = pd.DataFrame(alltime[:, np.newaxis], columns=['Time'])
            dataframe = pd.concat([dt, df, Time], axis=1)
        #　np.savetxt(allpath, df, delimiter=',', fmt='%d')
        df.to_csv(allpath)
##########################################################################

