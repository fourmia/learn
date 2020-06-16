import numpy as np
import pandas as pd
import xml.etree.cElementTree as ET
import re
import os
from Dataprocess import Znwg, Writefile


def regex(path):
    # 正则表达式确定文件名称
    #today = Znwg.znwgtime()
    today = '20200324080000'
    pattern = r'(\w+'+today+'.*?.csv)'
    pattern = re.compile(pattern)
    te = ','.join(os.listdir(path))
    fnames = re.findall(pattern, te)
    return fnames
'''
def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
        # return string
    return str1
'''

def readIndex(path):
    """
    从不同的路径中读取指数文件，几个路径则表明返回几个dataframe
    :return: 含Dataframe的list-->[Dataframe, Dataframe ...]
    """
    allindexpath = Znwg.regex(path, 6)
    '''
    tree = ET.parse(path)
    root = tree.getroot()
    allindexpath = [i.text for i in root[-1]]  # 这个应该仅确定目录，通过正则来确定具体的文件名参数
    '''
    print(allindexpath)
    '''
    allfname = []    # allfname应返回多个列表
    for i in range(len(allindexpath)):
        fnames = regex(allindexpath[i])
        allfname.append(fnames)
    '''
    allfname = [regex(index) for index in allindexpath]
    print(allfname)
    windpath, icepath, floodpath = [], [], []
    windvalue, icevalue, floodvalue = [], [], []
    for i in range(len(allfname[0])):
        windpath.append(os.path.join(allindexpath[0], allfname[0][i]))
        windvalue.append(pd.read_csv(os.path.join(allindexpath[0], allfname[0][i])))
        icepath.append(os.path.join(allindexpath[1], allfname[1][i]))
        icevalue.append(pd.read_csv(os.path.join(allindexpath[1], allfname[1][i])))
        floodpath.append(os.path.join(allindexpath[2], allfname[2][i]))
        floodvalue.append(pd.read_csv(os.path.join(allindexpath[2], allfname[2][i])))
    return windvalue, icevalue, floodvalue


def clcindex(data, path):
    indexpath = Writefile.readxml(path, 6)
    trafficindex = [np.max(data[i], axis=0) for i in range(56)]
    fname = ['%03d' % i for i in range(3, 169, 3)]
    filetime = Znwg.znwgtime()
    Writefile.write_to_csv(indexpath, trafficindex, 'trafficindex', fname, filetime)


def main():
    path = r'/home/cqkj/project/Product/Product/config/traffic.xml'
    data = readIndex()
    clcindex(data, path)
    '''
    path = r'/home/cqkj/QHTraffic/Product/Product/config/Traffic.xml'
    data = readIndex(path)
    '''


if __name__ == '__main__':
    main()
