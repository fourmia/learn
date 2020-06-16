import os
import pickle
import numpy as np
import datetime as dt
import glovar
from Dataprocess import interp, ecmwf, Writefile, Znwg
from DataInterface import Datainterface


def presnow():
    # 得到前一时刻积雪深度
    ectime = ecmwf.ecreptime()
    fh = [0]
    dic = 'ECMWF_HR/SNOD'
    lon, lat, data = Datainterface.micapsdata(ectime, dic, fh)
    return interp.interpolateGridData(data, lat, lon, glovar.lat, glovar.lon)


def snowData():
    # 获取ec数据信息(气温、降水、地温、湿度、积雪深度)
    ectime = ecmwf.ecreptime()
    fh = [i for i in range(12, 181, 3)]    # 20点的预报获取今天8:00的ec预报
    *_, dics = Writefile.readxml(glovar.trafficpath, 0)
    # *_, dics = Writefile.readxml(r'/home/cqkj/LZD/Product/Product/config/Traffic.xml', 0)
    dicslist = dics.split(',')[:-1]
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
        dataset.append(newdata)                     # 保存插值后的数据集
    return np.array(dataset)

def reverse(saltedata, dataset, snowdepth):
    """
    # 加载模型生成积雪深度模型结果
    :param saltedata: 卫星数据
    :param dataset:   ec气象要素数据
    :return:
    """
    tmp = [data.reshape(-1, 1) for data in dataset]  # 转换基础要素
    ele = np.concatenate(tmp, axis=1)
    ele.resize(56, 901 * 1401, 4)                              # 转换形状，将上一时刻积雪输入
    temp = np.nan_to_num(ele)
    snowdepth = snowdepth.reshape(-1, 1)  # 积雪深度数据，仅包含前一时刻
    m1, m2, savepath, roadpath, indexpath, _ = Writefile.readxml(glovar.trafficpath, 0)
    # m2 = r'/home/cqkj/LZD/Product/Product/Source/snow.pickle'
    if saltedata is not None:
        with open(m1, 'rb') as f:
            model1 = pickle.load(f)
            #########################################
        saltedata.resize(901 * 1401, 1)
        typecode = 1
    else:
        with open(m2, 'rb') as f:
            model2 = pickle.load(f)
        typecode = 2
    alldata = []
    ################################################
    for i in range(56):
        # temp = [data.reshape(-1, 1) for data in dataset[i]]  # 仅包含基础要素
        # newdataset = np.concatenate([temp, snowdepth, saltedata], axis=1)
        if typecode == 1:
            newdataset = np.concatenate([temp[i], snowdepth, saltedata], axis=1)
            prediction = np.array(model1.predict(newdataset))  # 每轮结果
        if typecode == 2:
            #print(presnow.shape)
            # 此处预报结果不可用图像呈现出分块
            newdataset = np.concatenate([temp[i], snowdepth], axis=1)
            prediction = np.array(model2.predict(np.nan_to_num(newdataset))) # 每轮结果
            # predictions = np.nan_to_num(model2.predict(np.nan_to_num(newdataset)))
            predictions = np.nan_to_num(prediction)
            # predictions[predictions < 0] = 0
            print(predictions.shape)
        snowdepth = predictions[:, np.newaxis]  # 结果作为下一次预测的输入
        predictions.resize(len(glovar.lat), len(glovar.lon))
        sdgrid = np.nan_to_num(predictions)
        sdgrid[sdgrid < 0] = 0
        alldata.append(sdgrid)
        sp = r'/home/cqkj/QHTraffic/Data//'
    Writefile.write_to_nc(sp, np.array(alldata), glovar.lat, glovar.lon, 'SnowDepth', glovar.fnames, glovar.filetime)
    return np.array(alldata)  # 返回 [56, 901, 1401]网格数据

def main():
    dataset = snowData()
    snowdepth = presnow()
    reverse(None, dataset, snowdepth)

if __name__ == '__main__':
    main()
