import pickle
import numpy as np
import xarray as xr
import datetime
import os
from collections import deque
import glovar
from DataInterface import Datainterface
from Dataprocess import Znwg, Writefile, ecmwf, interp, mkdirdate, insert_update


def snowdepth(path):
    # 6.18测试
    now = Znwg.znwgtime().strftime('%Y%m%d%H')
    pattern = r'(' + now + '.*?.nc)'
    strings = os.listdir(mkdirdate.dataMdir(path))
    namelist = sorted(Znwg.regex(pattern, strings))
    os.chdir(mkdirdate.dataMdir(path))
    datasets = xr.open_mfdataset(namelist, concat_dim='time')
    data = datasets.SnowDepth.values
    newdata = []
    for i in range(0, 7):
        #print(i * 8, (i + 1) * 8)
        tmp = np.mean(data[i * 8:(i + 1) * 8], axis=0)
        #print(tmp.shape)
        newdata.append(tmp)
    newdata.extend(newdata[-3:])
    newdata = np.array(newdata)
    newdata = np.piecewise(newdata, [newdata<0.1, newdata>=0.1], [1, 0])
    return newdata


def landtype(greenpath, forestpath):
    # 已测试
    green = xr.open_dataset(greenpath).green.values
    forest = xr.open_dataset(forestpath).forgest.values
    return green, forest


def liverain(path, pklpath):
    # ???????ZNWG???????????????pickle
    # ????????????????????????????
    elements, _, localdir, historydir, freq, *ftp = Writefile.readxml(path, 1)
    now = datetime.datetime.now()
    ytd = now - datetime.timedelta(days=1)
    dir = r'/SCMOC/BEXN'
    remote_url = os.path.join(dir, ytd.strftime('%Y'), ytd.strftime('%Y%m%d'))
    grb = Datainterface.GribData()
    grb.mirror('ER24', remote_url, localdir, ftp， freq= '24024')  # ??????????????
    rainpath = sorted(os.listdir(localdir))[-1]
    os.chdir(localdir)
    rainlive, lat, lon, res = Znwg.arrange([grb.readGrib(rainpath, nlat=glovar.latt, nlon=glovar.lonn)][0])
    ####??????????????????
    with open(pklpath, 'rb') as f:
        data = pickle.load(f)
    data.append(rainlive)
    # ????deque????
    with open(pklpath, 'wb') as f:
        pickle.dump(data, f)
    return rainlive


def Weatherdata(path):
    # ?????????????????????
    elements, subdirs, localdir, _, freq, *ftp = Writefile.readxml(path, 1)
    now = datetime.datetime.now()
    elements = elements.split(',')
    subdirs = subdirs.split(',')
    remote_urls = [os.path.join(subdir, now.strftime('%Y'), now.strftime('%Y%m%d')) for subdir in subdirs]  # ??????

    grib = Datainterface.GribData()
    '''
    [grib.mirror(element, remote_url, localdir, ftp, freq=freq) for element, remote_url in
     zip(elements[:-1], remote_urls[:-1])]  # ???????????????????????????(24003)
     '''
    for element, remote_url in zip(elements[:-1], remote_urls[:-1]):
        grib.mirror(element, remote_url, localdir, ftp, freq=freq)

    grib.mirror(elements[-1], remote_urls[-1], localdir, ftp,  freq='24024')  # ???????????
    # ?????????????????????????????????????pattern
    strings = ','.join(os.listdir(localdir))
    patterns = [r'(\w+.EDA.*?.GRB2)', r'(\w+.ERH.*?.GRB2)', r'(\w+.TMP.*?.GRB2)', r'(\w+.ER24.*?.GRB2)']
    allpath = [localdir + sorted(Znwg.regex(pattern, strings), key=str.lower)[-1] for pattern in patterns] # allpath?????????????????????
    ele14list = slice(1, 74, 8)  # ??+2-1??????????????10?????14?????????
    ####????????wind????u???v??
    wind = grib.readGrib(allpath[0])[0]
    windu_v = np.array([v for _, v in wind.items()])
    windu, windv  = windu_v[::2][ele14list], windu_v[1::2][ele14list]
    data = np.array([Znwg.arrange(grib.readGrib(path))[0][ele14list] for path in allpath[1:-1]])  # ?????????????????
    #er, lat, lon, size = Znwg.arrange(grib.readGrib(allpath[-1], nlat=glovar.lat, nlon=glovar.lon))  # ???????????????????????????????????????
    er, lat, lon, size = Znwg.arrange([grib.readGrib(allpath[-1], nlat=glovar.latt, nlon=glovar.lonn)][0])
    result = windu, windv, data, er  # ??????????[4,10,181,277]????
    return result, lat, lon


def firelevel(data, path, snow, landtype):
    """
    ????????????
    :param data: ??????????????????
    :return: greenfirelevel
    """
    ############################################################
    # ??????????????????????????????????????????pickle
    with open(path, 'rb') as f:
        predq = pickle.load(f)
    preres = np.argmax(np.array(predq)[::-1], axis=0)
    tem = np.add.reduce(predq)
    preres[tem == 0] = 8                       # ?????????
    ##############################################################
    # ?????????????????????????????????????????? 0?? ????1??
    *eda, erh, tmp, er = data                   # eda?U??V??????????UV?????????????tmp???????????????????????
    refertest = []
    for i in range(len(er)):
        if i == 0:
            test = np.piecewise(er[i], [er[i] < 0.1, er[i] >= 0.1], [1, 0])  # ????????????????
            # np.piecewise(test, [test == 0, test > 0], [0, test+preres])
            test = np.where(test>0, test+preres, 0)
        else:
            test = np.argmax(er[:i + 1][::-1], axis=0)
            refer = np.add.reduce(er[:i + 1])
            test[refer == 0] = i + 1
            # np.piecewise(test, [test < i+1, test >= i+1], [test, lambda x:x+preres])
            test = np.where(test>=i+1, test+preres, test)
        refertest.append(test)
    #############################################################
    # ????????????????????????????
    eda = np.sqrt(eda[0]**2 + eda[1]**2)                                 # ????uv????????????
    edaindex = np.piecewise(eda, [(0<=eda)&(eda<1.6), (eda>=1.6)&(eda<3.5), (eda>=3.5)&
                                  (eda<5.6), (eda>=5.6)&(eda<8.1), (eda>=8.1)&(eda<10.9),
                                  (eda >=10.9)&(eda<14),(eda>14)&(eda<=17.2), eda>17.2],
                            [3.846, 7.692, 11.538, 15.382, 19.236, 23.076, 26.923, 30.9])

    tmp -= 273.15                                    # ???????????
    tmpindex = np.piecewise(tmp, [tmp<5, (tmp>=5)&(tmp<11), (tmp>=11)&
                                  (tmp<16), (tmp>=16)&(tmp<21), (tmp>=21)&(tmp<=25),
                                  tmp>25], [0, 4.61, 6.1, 9.23, 12.5, 15.384])

    erhindex = np.piecewise(erh, [erh>70, (erh>=60)&(erh<=70), (erh>=50)&
                                  (erh<60), (erh>=40)&(erh<50), (erh>=30)&(erh<=40),
                                  erh<30], [0, 3.076, 6.153, 9.23, 12.307, 15.384])
    refertest = np.array(refertest)
    mindex = np.piecewise(refertest, [refertest==0, refertest==1, refertest==2, refertest==3,
                                      refertest==4, refertest==5, refertest==6, refertest==7,
                                      refertest>=8], [0, 7.692, 11.538, 19.23, 23.076, 26.923,
                                                      30.7, 34.615, 38])
    u = edaindex + tmpindex + erhindex + mindex
    rain = np.piecewise(er, [er<0.1, er>=0.1], [0, 1])
    rain = [interp.interpolateGridData(r, glovar.latt, glovar.lonn, glovar.lat, glovar.lon) for r in rain]
    rain = np.nan_to_num(np.array(rain))

    u = [interp.interpolateGridData(u_, glovar.latt, glovar.lonn, glovar.lat, glovar.lon) for u_ in u]
    u = np.nan_to_num(np.array(u))
    green = u*landtype[0]*rain*snow                                 #  ?????????
    forest = u*landtype[1]*rain*snow                                #  ????????
    ###################################################################################
    # ????????????????????
    gindex = np.piecewise(green, [green<=25, (green>25)&(green<51), (green>=51)&(green<73), (green>=73)&(green<91), green>=91], [1,2,3,4,5])
    findex = np.piecewise(forest, [forest <= 25, (forest > 25) & (forest < 51), (forest >= 51) & (forest < 73),
                                  (forest >= 73) & (forest < 91), forest >= 91], [1, 2, 3, 4, 5])
    mindex = np.maximum(gindex, findex)                             # 气象火险m

    return gindex, findex, mindex


def main():
    snowpath, gpath, fpath, rainpath, fspath, gspath, mspath = Writefile.readxml(glovar.forestpath, 0)                       # ???nc??????????
    snow = snowdepth(snowpath)           # ???????[10, 801, 1381]
    data, *_ = Weatherdata(glovar.forestpath)   # ??????????
    ldtype = landtype(gpath, fpath)
    gindex, findex, mindex = firelevel(data, rainpath, snow, ldtype)  # ???????????
    filetime = ecmwf.ecreptime()
    fh = range(10)
    fnames = ['_%03d' % i for i in fh]
    Writefile.write_to_nc(gspath, gindex, filetime=filetime, fnames=fnames, lat=glovar.lat, lon=glovar.lon,
                          name='green', elename=None)
    Writefile.write_to_nc(fspath, findex, filetime=filetime, fnames=fnames, lat=glovar.lat, lon=glovar.lon,
                          name='forest', elename=None)

    Writefile.write_to_nc(mspath, mindex, filetime=filetime, fnames=fnames, lat=glovar.lat, lon=glovar.lon,
                          name='meteo', nums= 1, elename='risk')
if __name__ == '__main__':
    main()




