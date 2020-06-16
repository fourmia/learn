import os
import sys
import numpy as np
import torch
import datetime
import glovar
import pandas as pd
from Model.rainrevisenet import LeNet    # 此后训练模型应先用字典保存模型信息
from Dataprocess import Writefile, Znwg, dataformat, interp
from DataInterface import Datainterface


def rainData():
    # 同步降雨智能网格文件并解析
    now = datetime.datetime.now()
    # *_, elements, ftp = Writefile.readxml(glovar.trafficpath, 1)
    *_, elements, ftp = Writefile.readxml(glovar.trafficpath, 5)
    element = elements.split(',')
    ftp = ftp.split(',')
    grib = Datainterface.GribData()
    remote_url = os.path.join(r'\\SPCC\\BEXN', now.strftime('%Y'), now.strftime('%Y%m%d'))
    grib.mirror(element[0], remote_url, element[1], ftp, element[2])
    rname = sorted(os.listdir(element[1]))[-1]
    rpath = element[1] + rname
    dataset, lat, lon, _ = Znwg.arrange((grib.readGrib(rpath)))    # result包含data,lat,lon,size
    return np.array([interp.interpolateGridData(data, lat, lon, glovar.lat, glovar.lon) for data in dataset[:56]])


def revise(message):
    # 订正解析结果，并插值到1km分辨率
    mdpath, gcpath, savepath, *_ = Writefile.readxml(glovar.trafficpath, 1)
    net = torch.load(mdpath)
    dem = pd.read_csv(gcpath, index_col=0).values
    arrays = np.array([np.nan_to_num([data, dem]) for data in message[:, :801, :1381]])
    inputs = torch.from_numpy(arrays)
    # torch.no_grad()
    outputs = [net(it[np.newaxis, :]).detach().numpy() for it in inputs]
    outputs = np.nan_to_num(outputs)
    outputs[outputs < 0] = 0
    print(outputs.shape)
    output = np.squeeze(outputs)
    lat = np.linspace(31.4, 39.4, 801)
    lon = np.linspace(89.3, 103.1, 1381)
    raingb = np.array([np.nan_to_num(interp.interpolateGridData(op, lat, lon, glovar.lat, glovar.lon)) for op in message])        # 后续对模型更改
    Writefile.write_to_nc(savepath, raingb, glovar.lat, glovar.lon, 'Rain', glovar.fnames, glovar.filetime)
    return outputs


def main():
    message = rainData()
    revise(message)
    return None


if __name__ == '__main__':
    main()






