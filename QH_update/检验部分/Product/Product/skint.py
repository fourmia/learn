import os
import pickle
import numpy as np
import datetime as dt
import glovar
from sklearn.preprocessing import StandardScaler
from Dataprocess import interp, ecmwf, Writefile, Znwg
from DataInterface import Datainterface


def skintData():
    # 获取ec数据信息(气温、降水、地温、湿度、积雪深度)
    ectime = ecmwf.ecreptime()
    fh = [i for i in range(12, 181, 3)]    # 20点的预报获取今天8:00的ec预报
    # *_, dics = Writefile.readxml(glovar.trafficpath, 0)
    dics = Writefile.readxml(glovar.trafficpath, 3)[3]
    print(dics)
    dicslist = dics.split(',')
    lonlatset, dataset = [], []
    for dic in dicslist:
        newdata = []
        lon, lat, data = Datainterface.micapsdata(ectime, dic, fh)
        lonlatset.append((lon, lat))
        for i in range(data.shape[0] - 1):
            if (np.isnan(data[i]).all() == True) and (i + 1 <= data.shape[0]):
                data[i] = data[i + 1] / 2
                data[i+1] = data[i + 1] / 2
                newdata.append(interp.interpolateGridData(data[i], lat, lon, glovar.lat, glovar.lon))
            else:
                newdata.append(interp.interpolateGridData(data[i], lat, lon, glovar.lat, glovar.lon))
        newdata = np.array(newdata)
        # newdata[newdata<0] = 0                    # 保证数据正确性
        dataset.append(newdata)                     # 保存插值后的数据集(要素包括气温、相对湿度、降水、地温)
    return np.array(dataset)


def reverse(dataset):
    # 此函数用来生成地表最高、 最低温度产品，对现有模型进行整体改动
    maxpath, minpath, *_ = Writefile.readxml(glovar.trafficpath, 3)
    with open(maxpath, 'rb') as f:
        maxmodel = pickle.load(f)
    with open(minpath, 'rb') as f:
        minmodel = pickle.load(f)
    # dataset.resize(56, 801 * 1381, 4)
    ################################################
    temp = [data.reshape(-1, 1) for data in dataset]   # 数据量可能过大
    allele = np.concatenate(temp, axis=1) #  训练用元素
    # 先采用一个进行测试
    maxvalue = maxmodel.predict(allele).reshape(56, 901, 1401)
    minvalue = minmodel.predict(allele).reshape(56, 901, 1401)
    # 路径
    savepath= r'/home/cqkj/QHtraffic/Data/skint//'
    Writefile.write_to_nc(savepath, maxvalue, glovar.lat, glovar.lon, 'maxskint', glovar.fnames, glovar.filetime)
    Writefile.write_to_nc(savepath, minvalue, glovar.lat, glovar.lon, 'minskint', glovar.fnames, glovar.filetime)
    return maxvalue, minvalue                         # 返回地表最高、最低温度



def main():
   mes = skintData()
   reverse(mes) 


if __name__ == "__main__":
    main()
