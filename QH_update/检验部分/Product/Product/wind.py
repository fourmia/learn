import pickle
import torch
import os
import numpy as np
import glovar
from Dataprocess import interp, Znwg, Writefile, dataformat
from DataInterface import Datainterface
from Model.lenet import LeNet
import pandas as pd



def windData(path):
    # 获取数据信息
    *_, elements, ftp = Writefile.readxml(path, 2)
    element = elements.split(',')
    ftp = ftp.split(',')
    grib = Datainterface.GribData()
    remote_url = os.path.join(r'\\SPCC\\BEXN', glovar.now.strftime('%Y'), glovar.now.strftime('%Y%m%d'))
    grib.mirror(element[0], remote_url, element[1], ftp, element[2])
    rname = sorted(os.listdir(element[1]))[-1]
    rpath = element[1] + rname
    dataset, lat, lon, _ = Znwg.arrange((grib.readGrib(rpath)))                       # result包含data,lat,lon,size
    return [interp.interpolateGridData(data, lat, lon, glovar.lat, glovar.lon) for data in dataset]      # 返回插值后列表格式数据

def revise(path, message):
    # 订正解析结果，并插值到1km分辨率
    mdpath, _, gcpath, savepath, indexpath, *_ = Writefile.readxml(path, 2)
    data = np.array(message)
    data = [np.nan_to_num(data[::2, :, :][:56]), np.nan_to_num(data[1::2, :, :][:56])]
    net = torch.load(mdpath)
    net.eval()
    dem = pd.read_csv(gcpath, index_col=0).values
    arrays = np.array([np.array([i, j, dem]) for i, j in zip(data[0][:,:801,:1381], data[1][:,:801,:1381])])
    inputs = torch.from_numpy(arrays)
    torch.no_grad()
    outputs = [np.nan_to_num(net(it[np.newaxis, :]).detach().numpy()) for it in inputs]
    datau, datav = np.squeeze(outputs)[:,0,...], np.squeeze(outputs)[:,1,...]
    # 统一格式    
    lat = np.linspace(31.4, 39.4, 801)
    lon = np.linspace(89.3, 103.1, 1381)
    uwind = [np.nan_to_num(interp.interpolateGridData(u, lat, lon, glovar.lat, glovar.lon)) for u in datau]
    vwind = [np.nan_to_num(interp.interpolateGridData(v, lat, lon, glovar.lat, glovar.lon)) for v in datav]


    Writefile.write_to_nc(savepath, np.array(uwind), glovar.lat, glovar.lon, 'U', glovar.fnames, glovar.filetime)
    Writefile.write_to_nc(savepath, np.array(vwind), glovar.lat, glovar.lon, 'V', glovar.fnames, glovar.filetime)
    return indexpath, datau, datav


def press(savepath, datau, datav):
    # 计算出道路点风压大小
    mes = pd.read_csv(glovar.windpath, index_col=0)
    lon, lat = mes['Lon'], mes['Lat']  # 提取出道路坐标
    length = np.linalg.norm(mes.iloc[:, -2:], axis=1)
    ur = np.divide(mes['ur'], length)
    vr = np.divide(mes['vr'], length)
    uv = [[i, j] for i, j in zip(ur, vr)]
    latt, lonn = np.linspace(31.4, 39.4, 801), np.linspace(89.3, 103.1, 1381)
    for u, v in zip(datau, datav):
        uvalue = interp.interpolateGridData(u, latt, lonn, lat, lon, isGrid=False)
        vvalue = interp.interpolateGridData(v, latt, lonn, lat, lon, isGrid=False)
        UVvalue = [[i, j] for i, j in zip(uvalue, vvalue)]
        windcross = []
        for i,value in enumerate(UVvalue):
            datas = np.cross(uv, np.array(value).T)
            w = 1 / 2 * 1.29 * (np.square(datas)) * 1000
            w = np.piecewise(w, [w < 83, (w >= 83) & (w < 134), (w >= 134) & (w < 602), (w >= 602) & (w < 920), w >= 920],
                              [1, 2, 3, 4, 5])
            windcross.append(w[np.newaxis, :])
    Writefile.write_to_csv(savepath, windcross, 'windindex', glovar.fnames, glovar.filetime)


def main():
   message = windData(glovar.trafficpath)
   savepath, u, v = revise(glovar.trafficpath, message)
   press(savepath, u, v)
   return None

if __name__ == '__main__':
    main()

