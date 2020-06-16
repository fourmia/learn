import datetime as dt
import numpy as np
from functools import wraps
"""此模块主要用于M4中ecmwf数据后处理
目前包含EC文件时间，EC范围缩减处理
"""

def ecreptime():
    """获取当前时间上一个08时或20时（就近）,用于EC
    取EC滞后半天的数据用于预报"""
    now = dt.datetime.now()
    if now.strftime('%H') >= '00' and now.strftime('%H') < '16':
        temp = now.replace(hour=20, minute=0, second=0)
        report = temp - dt.timedelta(days=1)
    elif now.strftime('%H') >= '16':
        report = now.replace(hour=8, minute=0, second=0)
    print(report)
    return report.strftime('%Y%m%d%H%M')


def interceptarea(region):
    """
    此装饰器主要是为了在M4大数据范围内得到指定范围的数据，这仅仅是一个初始版本，必须确保
    待装饰函数返回lon，lat，data（data应为ndarray数组）
    :param region:需要截取的经纬度范围 规定为(lonmin, latmin, lonmax, latmax)
    :return: region 内部的data
    """
    def decorate(func):
        """
        接受上层函数传递的经纬度范围
        :param func:func为待装饰函数
        """
        f_lon, l_lon = region[0] - 2, region[2] + 2
        f_lat, l_lat = region[1] - 2, region[3] + 2
        @wraps(func)
        def wapper(*args, **kwargs):
            lons, lats, datas = func(*args, **kwargs)
            # 得到边界索引--->更好的实现方法后续采用len实现
            idx_llon = np.where(lons == lons[lons > f_lon][0])[0]
            idx_rlon = np.where(lons == lons[lons < l_lon][-1])[0]
            idx_dlat = np.where(lats == lats[lats > f_lat][-1])[0]
            idx_ulat = np.where(lats == lats[lats < l_lat][0])[0]
            lons = lons[idx_llon[0]: idx_rlon[0]]
            lats = lats[idx_ulat[0]: idx_dlat[0]]
            #print(datas.shape)
            newdata = [data[idx_ulat[0]: idx_dlat[0], idx_llon[0]: idx_rlon[0]] for data in datas]
            newdata = np.array(newdata)
            #print("idx_dlat:{}, idx_ulat:{}".format(idx_dlat, idx_ulat))
            #print("idx_llon:{}, idx_rlon:{}".format(idx_llon, idx_rlon))
            #print("lon:{}-{}, lat:{}-{}".format(lons[0], lons[-1], lats[-1], lats[0]))
            #print(newdata.shape)
            return lons, lats, newdata
        return wapper
    return decorate
