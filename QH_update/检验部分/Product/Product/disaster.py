import os
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
from Product import glovar
from Product.Dataprocess import Writefile, ecmwf, Znwg


def filelist(path):
    now = dt.datetime.now().strftime('%Y%m%d')
    pattern = r'('+ now + ')'
    rpath, spath, wpath, icpath, savepath = Writefile.readxml(path, 2)
    pathlists = [rpath, spath, wpath, icpath]
    # 获取出四个智能网格数据列表
    elements = [Znwg.regex(pattern, os.listdir(path)) for path in pathlists]
    return elements


def main():
    # 计算出来灾害落区
    # dataset 为获取的各气象要素等级预报
    ######读取nc要素文件#####
    varname = ['Rain', 'Snow_depth', 'Wind', 'Roadic']
    datasets = [xr.open_mfdataset(path, concat_dim='time').values for path,name in zip(filelist(), varname)]
    rain, snow, wind, roadic = datasets
    roadic *= 8
    snow *= 4
    wind *= 2
    rain *= 1
    ########################
    disaster = roadic + snow + wind + rain
    ########################
    configpath = r'../config/disaster.xml'
    savepath = Writefile.readxml(configpath, 1)[-1]
    filetime = ecmwf.ecreptime()
    fh = range(3, 169, 3)
    fnames = ['_%03d' % i for i in fh]
    Writefile.write_to_nc(savepath, disaster, glovar.lat, glovar.lon, 'Disaster', fnames, filetime)


if __name__ == '__main__':
    main()
