# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:52:54 2019
https://firms.modaps.eosdis.nasa.gov/active_fire/c6/text/MODIS_C6_Russia_and_Asia_24h.csv
@author: YchZhu
"""
# 导入数据库
import os
import time
import requests
#from you_get import common
import csv
import pandas as pd
# import os
from retrying import retry
from netrc import netrc


@retry(stop_max_attempt_number=10)
def spiders():
    # 待下载数据的下载链接
    f = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Russia_Asia_24h.csv"

    # 数据的存在位置
    saveDir = r'/home/cqkj/QHTraffic/Data/MODIS//'
    # 数据的保存名称：和数据下载名称一致

    saveName =saveDir + f.split('/')[-1].strip()

    print(saveName)
    '''
    common.any_download(url=f, output_dir=saveDir)
    cmd = r'mv C:\\Users\GJW\Desktop\MODIS\MODIS_C6_Russia_Asia_24h.None C:\\Users\GJW\Desktop\MODIS\MODIS_C6_Russia_Asia_7d.csv'
    os.system(cmd)
    '''
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
        '''
        if response.status_code != 200:
            print("{} not downloaded. Verify that your username and password are correct in {}")
            raise
        else:
        '''
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


if __name__ == '__main__':
    spiders()
