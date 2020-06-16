import os
import traceback
import numpy as np
import pandas as pd
from . import DataBlock_pb2
import datetime as dt
from . import GDS_data_service
from .read_Data import read
from Dataprocess import ecmwf, Znwg, interp
import pygrib
#from cma.cimiss.DataQueryClient import DataQuery
# tiff处理
#from osgeo import gdal, osr


#@interceptarea(region=(110.21, 31.23, 116.39, 36.22))
def micapsdata(ectime, dic, fh, IP="10.69.72.112", port=8080):
    """
    micapsdata方法主要返回不经过任何处理的M4数据
    :param dir: 待获取的数据要素目录，如 'ECMWF_HR/TMP_2M'
    :param fh: 要获取的数据要素的文件有效数字后缀， 如‘3’,'6', '9'......
    :return:  不激活装饰器，返回不经任何处理的M4原始数据
               激活装饰器，返回指定范围的数据，程序运行过慢
    """
    it = dt.datetime.strptime(ectime, '%Y%m%d%H%M')   # 获取特定时间文件
    t0 = dt.datetime.strftime(it, '%y%m%d%H')
    fnames = [t0 + '.%03d' % i for i in fh]
    # nd  = np.array([])
    nds = []
    fts = []
    # 解析数据
    service = GDS_data_service.GDSDataService(IP, port)
    status, response = service.getFileList(dic)
    MappingResult = DataBlock_pb2.MapResult()
    # 获取文件名列表
    if status == 200 and MappingResult is not None:
        MappingResult.ParseFromString(response)
        results = MappingResult.resultMap
        names = [name for name,_ in results.items()]
        # names = (name for name,_ in results.items())
        # print('++'*10)
        # print(results)
        print(names)
        print(fnames)
        for f in fnames:                      # fnames 为自己构造的名称
            if f in names:
                _, response = service.getData(dic, f)
                rd = read(response)
                ft = rd.time + dt.timedelta(hours=rd.period)
                print(ft)
                # nd_ = [data for data in rd.data]
                # nd = np.array(nd_)
                nd = np.array([data for data in rd.data])
                # test nd
                print(nd.shape)
                nds.append(nd)
                fts.append(ft)
    # 对此处还需要进行测试
    new_nds = []
    all_times = [it + dt.timedelta(hours=i) for i in fh]
    i = 0
    print('*' * 10 + 'all_times' + '*' * 10)
    print(all_times)
    print('*' * 10 + 'fts' + '*' * 10)
    print(fts)
    for t in all_times:                               # 所需的所有时间
        if t in fts and i < len(all_times)-1:                                  # 目前有的时间
            new_nds.append(nds[i])
            i = i + 1
        else:                                         # 逐六中间缺少的逐三
            #new_nds.append(nds[i+1]/2)
            new_nds.append(np.full(nd.shape, np.nan))
    new_nds = np.concatenate(new_nds, axis=0)
    print('*'*10 + 'New_nds shape' +'*'*10)
    print(new_nds.shape)
    if new_nds.size == 0:
        micapsdata(dic, fh)
    print('lon_max:{}, lon_min:{}, lat_max:{}, lat_min:{}, res:{}'.format(
        rd.lon[-1], rd.lat[0], rd.lat[0], rd.lat[-1], rd.lon[1]-rd.lon[0]
    ))

    return rd.lon, rd.lat, new_nds

def cimissdata(interfaceId, *args, **kwargs):
    """
    此函数用于获取cimiss数据，主要接受参数interfaceId，elements，和一个关键字参数，关键字参数对应该资料所需的必选params。
    注：在使用该函数获取资料时，默认最后为要素时间信息，对时间信息进行了相应处理，elements格式应为"XXXXX,Year,Mon,Day,Hour"
    仅处理到小时信息，需要分钟信息需自行在函数中添加
    params：
            interfaceId:接口Id
            args       :待获取的资料要素，格式为"XXXXX,Year,Mon,Day,Hour"，XXXX为所需获取的资料
            kwargs     :构造的params
    return:
            Dataframe:指定时间段的Dataframe,Dataframe后两列均为时间信息(Time, month)-->(2019/11/11/08, 11)
    """
    # 用户名和密码
    userId = "BEXN_HD_HHF"
    pwd = "898600"
    # 开始写接口信息,考虑要素的传入
    client = DataQuery()
    params = kwargs
    print(params)
    # print(args[0])
    try:
        result = client.callAPI_to_array2D(userId, pwd, interfaceId, params)
    finally:
        client.destroy()
    print("Get data!")
    print(111111111111111111111111111)
    print(args)
    print(111111111111111111111111111)
    cols = args[0].split(',')[0:-4] + ["year", "month", 'day', 'hour']
    print(cols)
    data = pd.DataFrame(result.data, columns=cols)
    data['Time'] = pd.to_datetime(data[['year', "month", 'day', 'hour']])
    # 转化为北京时间
    data['Time'] += dt.timedelta(hours=8)
    df = data.drop(columns=['year', 'day', 'month', 'hour'])
    print(df.columns)
    print(df.dtypes)
    # df[df.columns[8:-1]] = df[df.columns[1:-1]].astype('float')
    # print(df.datetimeypes)
    if df.empty and result.request.errorCode != -1:
        errorC = result.request.errorCode
        errorM = result.request.errorMessage
        raise Exception('errorCode:{}, errorMessage{}'.format(errorC, errorM))
    else:
        print("Get Data Success!")
        df["Time"] = pd.to_datetime(df['Time'])
        # print(df.Min.value_counts())
    return df


class GribData(object):
    def __init__(self):
        '''
        params:
                path   : str,READ isdir-Batch file;isfile-One file only
                isbz2   : False(defult,True-bunzip2()) ,Is it compressed file
                FH     : list
                islocal : True(defult,False-Sync to local) ,Is it a remote file
                ftp_url、user、password、remote_url : str,Remote information
                zipDir   : str,zip path
                bunzipDir  : str,bunzip path  ==path
                varname  :list(default=None), choose read varname,if varname=None,read all varname
        '''
        pass


    def mirror(self, element, remote_url, localdir, *args, freq=None):
        """
        从服务器上同步数据,此处存在问题
        :param element: 需获取的元素名称 eg：ER03、TMAX
        :param path: 远程服务器下文件的路径信息，用以构造remote_url
        :param localdir: 本机用来存放同步的grib文件目录
        :param freq: 同步文件的时间分辨率信息，eg24003，24024
        :param args: 服务器名称、用户名、密码
        :return:
        """
        print(*args)
        ftp_url, user, password = args[0]
        print(ftp_url, user, password)
        initnal_time = Znwg.znwgtime()
        print(initnal_time)
        if freq:
            cmd = '''lftp -c "open {ftp} -u {user},{password}; lcd {localdir};
             cd {remote_url};mirror --no-recursion -I *{element}_{init_time:%Y%m%d%H%M}_{freq}.GRB2" '''.format(
                ftp=ftp_url, user=user, password=password, localdir=localdir, remote_url=remote_url,
                element=element, init_time=initnal_time, freq=freq)
        else:
            initnal_time = dt.datetime.now()
            cmd = '''lftp -c "open {ftp} -u {user},{password}; lcd {localdir};
             cd {remote_url};mirror --no-recursion -I *{element}-{init_time:%Y%m%d}*.GRB2" '''.format(
                ftp=ftp_url, user=user, password=password, localdir=localdir, remote_url=remote_url,
                element=element, init_time=initnal_time)
        print(cmd)
        cmd
        try:
            os.system(cmd)
        except:
            traceback.print_exc()

    def readGrib(self, path, varname=None, level=[0, 100], editionNum=None, nlat=None, nlon=None):
        '''
        params:
                path    :str,READ isdir-Batch file;isfile-One file only
                FH      :list/numpy.ndarray list Time effective list of documents to be processed eg:FH=np.arange(0,73,3)
                varname :list/numpy.ndarray  list of meteorological elements required
                level :[]  level range
                editionNum :grib encoding defulat=None  1 or 2
                nlat    :list/numpy.ndarray  Clip the range of latitude
                nlon    :list/numpy,ndarray  Clip the range of longitude
        return :
                data :dict  keys format='name_level_fh'
                lat  :list
                lon  :list
                size :list
        '''
        try:
            if not os.path.exists(path):
                raise ValueError('%s does not exist, please enter a valid path!' % path)
            grbs = pygrib.open(path)
            if editionNum is not None:
                grbs = grbs(editionNumber=editionNum)
            self.data, self.lat, self.lon, self.size, self.attrs = {}, [], [], [], {}
            if varname is None:
                for grb in grbs:
                    print(grb)
                    if (grb['level'] == 0) | ((grb['level'] > level[0]) & (grb['level'] < level[-1])):
                        self.init_time, name, dat, lat, lon, size = self.getInfo(path, grbs, grb, nlat, nlon)
                        self.data[name] = dat
                        self.attrs[grb.name] = grb.units
                        self.lat.append(lat)
                        self.lon.append(lon)
                        self.size.append(size)
            else:
                for var in varname:
                    grb = grbs.select(shortName=var)
                    if (grb['level'] == 0) | ((grb['level'] > level[0]) & (grb['level'] < level[-1])):
                        self.init_time, name, dat, lat, lon, size = self.getInfo(path, grbs, grb, nlat, nlon)
                        self.data[name] = dat
                        self.attrs[grb.name] = grb.units
                        self.lat.append(lat)
                        self.lon.append(lon)
                        self.size.append(size)
            return self.data, np.array(self.lat), np.array(self.lon), np.array(self.size)
        except:
            traceback.print_exc()

    def getInfo(self, path, grbs, grb, nlat=None, nlon=None):
        '''
        :return: name, data, lat, lon, size, fh
        '''
        init_time = dt.datetime(grb.year, grb.month, grb.day,
                                      grb.hour, grb.minute, grb.second)
        fh = grb.stepRange
        print(fh)
        print(111111111111111111111111)
        if '-' in fh:
            fh = [int(i) for i in fh.split('-')][-1]
        else:
            fh = int(fh)
        # types=39->bcsh  Beijing->Z-  European->ec  US National Weather Service - NCEP->gfs
        # BIN
        types = grb.centreDescription
        print(types)
        if types == '39':
            # [Nj,Ni]
            if 'warr' in path:
                size = 0.03
            elif 'warms' in path:
                size = 0.09
            lat_ = grbs[16].values
            lon_ = grbs[17].values
            lat = np.arange(lat_.min(), lat_.max() + size / 2, size)
            lon = np.arange(lon_.min(), lon_.max() + size / 2, size)
            data = interp.interpolateGridData(grb.values, lat_, lon_, lat, lon, method='griddate',
                                       Size=size * 2, isGrid=True)
        else:
            # ec='European Centre for Medium-Range Weather Forecasts',z_='Beijing'
            # lat,lon= grb.latlons()   #ndim=2
            lat = grb.distinctLatitudes  # ndim=1
            lon = grb.distinctLongitudes  # ndim=1
            if grb.iDirectionIncrementInDegrees != grb.jDirectionIncrementInDegrees:
                raise ValueError('lat_size!=lon_size')
            size = grb.iDirectionIncrementInDegrees
            data = grb.values

        for i in [grb.shortName, grb.name, grb.parameterName]:
            name = i
            print('i is :{}'.format(i))
            if name == 'unknown':
                continue
            else:
                break
        # 智能网格
        if (name == 'unknown') & ('Beijing' in types):
            name = path.split('_')[-3].split('-')[-1]
            print('path:{}'.format(path))
        if name == 'unknown':
            raise ValueError('name is unknown')

        if grb['level'] > 0:
            name = name + '_' + str(grb['level'])
        name = name + '_' + str(fh)
        # print('lat:{}, lon:{}'.format(nlat, nlon))
        # 选择一定范围经纬度的值
        if (nlat is not None) | (nlon is not None):
            if nlat is None:
                nlat = lat
            if nlon is None:
                nlon = lon
            # print('#*20')
            # print('lat:{}, lon:{}'.format(nlat, nlon))
            if types == '39':
                ind_lat = np.where((lat >= nlat.min()) & (lat <= nlat.max()))[0]

                ind_lon = np.where((lon >= nlon.min()) & (lon <= nlon.max()))[0]
                data = data[ind_lat, :]
                data = data[:, ind_lon]
                lat, lon = lat[ind_lat], lon[ind_lon]
            else:
                print("This is a test")
                # print(grb.data())
                data, lats, lons = grb.data(lat1=nlat.min(), lat2=nlat.max(), lon1=nlon.min(), lon2=nlon.max())
                lat = lats[:, 0]
                lon = lons[0, :]

        return init_time, name, data, lat, lon, size

'''
class ReadTiff(object):
    """
    这个类主要用于读取tiff数据，并将图上坐标转化为地理坐标
    """
    def __init__(self, dataset):
        # path is tiff file path
        self._dataset = dataset
        self._shape = self._dataset.ReadAsArray().shape
        self.prosrs, self.geosrs = self.getSRSPair()

    def getSRSPair(self):
        """获取投影坐标系和地理坐标系"""
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(self._dataset.GetProjection())
        geosrs = prosrs.CloneGeogCS()
        return prosrs, geosrs

    def imagexy2pro(self):
        """将图上坐标转化为投影坐标"""
        row, col = np.mgrid[0:self._shape[0], 0:self._shape[1]]
        trans = self._dataset.GetGeoTransform()
        px = trans[0] + col * trans[1] + row * trans[2]
        py = trans[3] + col * trans[4] + row * trans[5]
        return px, py

    def pro2geo(self,x,y):
        """将投影坐标转化为地理坐标,仅仅适用于单点，非等经纬度"""
        ct = osr.CoordinateTransformation(self.prosrs, self.geosrs)
        coords = ct.TransformPoint(x,y)
        return coords[:2]
'''
if __name__ == '__main__':
    ectime = ecmwf.ecreptime()
    dics = ['ECMWF_HR/RH/1000', 'ECMWF_HR/SKINT']
    fh = list(range(12, 72, 3)) + list(range(72, 205, 6))
    data = micapsdata(ectime, dics, fh)
    with open('ectest.pkl', 'wb') as f:
        f.dump()
