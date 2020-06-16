import datetime as dt
import xml.etree.cElementTree as ET
import pymysql
import pandas as pd
import xarray as xr
import numpy as np
import glovar


def readxml(path, num):
    tree = ET.parse(path)
    root = tree.getroot()
    settings = [i.text for i in root[int(num)]]
    return settings


def readbs():
    host,uid,pwd,dbname,port,table_name = readxml()[1]
    # 有密码版本
    # conn = pymysql.connect(host=host, port=int(port), user=uid,
    #                   passwd=pwd, database=dbname)
    # 无密码版本
    conn = pymysql.connect(host=host, port=int(port), user=uid,
                       database=dbname)
    df = pd.read_sql('SELECT * FROM ' + table_name, conn)
    return df


def write_to_nc(path, data, lat, lon, name, fnames, filetime):
    cirnum = data.shape[0]
    print(cirnum)
    for i in range(cirnum):
        print(data[i][np.newaxis,:].shape)
        #print(dt.datetime.strptime(filetime,'%Y%m%d%H%M'))
        ds = xr.Dataset({name: (['time', 'lat', 'lon'], data[i][np.newaxis,:])}, coords={'time':[dt.datetime.strptime(filetime,'%Y%m%d%H%M')],'lat': lat, 'lon': lon})
        f = ''.join([path, filetime, '.', fnames[i], '.', name, '.nc'])
        ds.to_netcdf(f, format='NETCDF3_CLASSIC')


def write_to_csv(path, data, name, fnames, filetime):
    # 此处需添加road文件
    cirnum = len(data)
    # 读取road文件合并
    road = pd.read_csv(glovar.roadpath, index_col=0)
    for i in range(cirnum):
        # allpath = path + name + '_' + filetime + fnames[i] + '.csv'
        allpath = path + filetime + '.' + fnames[i] + '.' + name + '.csv'
        df = pd.concat([road, pd.DataFrame(data[i], columns=name, dtype=np.int8)])
        #　np.savetxt(allpath, df, delimiter=',', fmt='%d')
        df.to_csv(allpath)
