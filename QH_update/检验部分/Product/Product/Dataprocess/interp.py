from scipy import interpolate
import numpy as np
from scipy.spatial import cKDTree

"""此模块主要用于插值操作，包括站点插值及网格插值
"""
class tree(object):
    """
    Compute the score of query points based on the scores of their k-nearest neighbours,
    weighted by the inverse of their distances.

    @reference:
    https://en.wikipedia.org/wiki/Inverse_distance_weighting

    Arguments:
    ----------
        X: (N, d) ndarray
            Coordinates of N sample points in a d-dimensional space.
        z: (N,) ndarray
            Corresponding scores.
        leafsize: int (default 10)
            Leafsize of KD-tree data structure;
            should be less than 20.

    Returns:
    --------
        tree instance: object

    Example:
    --------

    # 'train'
    idw_tree = tree(X1, z1)

    # 'test'
    spacing = np.linspace(-5., 5., 100)
    X2 = np.meshgrid(spacing, spacing)
    X2 = np.reshape(X2, (2, -1)).T
    z2 = idw_tree(X2)

    See also:
    ---------
    demo()

    """

    def __init__(self, X=None, z=None, leafsize=10):
        if not X is None:
            self.tree = cKDTree(X, leafsize=leafsize)
        if not z is None:
            self.z = z

    def fit(self, X=None, z=None, leafsize=10):
        """
        Instantiate KDtree for fast query of k-nearest neighbour distances.

        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N sample points in a d-dimensional space.
            z: (N,) ndarray
                Corresponding scores.
            leafsize: int (default 10)
                Leafsize of KD-tree data structure;
                should be less than 20.

        Returns:
        --------
            idw_tree instance: object

        Notes:
        -------
        Wrapper around __init__().

        """
        return self.__init__(X, z, leafsize)

    def __call__(self, X, k=8, eps=1e-6, p=2, regularize_by=1e-9):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.

        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.

            k: int (default 8)
                Number of nearest neighbours to use.

            p: int or inf
                Which Minkowski p-norm to use.
                1 is the sum-of-absolute-values "Manhattan" distance
                2 is the usual Euclidean distance
                infinity is the maximum-coordinate-difference distance

            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.

            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.

        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.
        """
        self.distances, self.idx = self.tree.query(X, k, eps=eps, p=p)
        self.distances += regularize_by
        weights = self.z[self.idx.ravel()].reshape(self.idx.shape)
        mw = np.sum(weights / self.distances, axis=1) / np.sum(1. / self.distances, axis=1)
        return mw

    def transform(self, X, k=6, p=2, eps=1e-6, regularize_by=1e-9):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.

        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.

            k: int (default 6)
                Number of nearest neighbours to use.

            p: int or inf
                Which Minkowski p-norm to use.
                1 is the sum-of-absolute-values "Manhattan" distance
                2 is the usual Euclidean distance
                infinity is the maximum-coordinate-difference distance

            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.

            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.

        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.

        Notes:
        ------

        Wrapper around __call__().
        """
        return self.__call__(X, k, eps, p, regularize_by)



def interpolateGridData(data, lat, lon, newlat, newlon, method='griddate',
            Size=0.5, isGrid=True):
    '''
    params:
            data : numpy.ndarray , Raw data
            lat  : numpy.ndarray , Original latitude
            lon  : numpy.ndarray , Original longitude
            newlat : numpy.ndarray/float ,new latitude
            newlon : numpy.ndarray/float ,new longitude
            method : str (default=griddate),interpolate ('griddate','idw')
            Size   : float (default=0.5),Precision used to narrow query circles
            isGrib :bool （default=Yes）,Grid interpolation, or site
    return:
            newdata : numpy.ndarray/float
    '''
    if data.ndim >= 2:
        data = data.flatten()
        if lat.ndim ==1:
            lon, lat = np.meshgrid(lon, lat)
            lon = lon.flatten()
            lat = lat.flatten()
            
    if lat.ndim != lon.ndim:
        raise ValueError('lat ndim!=lon ndim')

    if lat.ndim >= 2:
        lat = lat.flatten()
        lon = lon.flatten()
        
    lonlat = np.array([lon, lat]).T
    # 缩小查询圈### 此处应为根据新坐标范围筛选出数据
    index = np.where((lonlat[:, 0] > (np.min(newlon) - Size)) & (lonlat[:, 0] < (np.max(newlon) + Size)) & \
                     (lonlat[:, 1] > (np.min(newlat) - Size)) & (lonlat[:, 1] < (np.max(newlat) + Size)))[0]
    lonlat = lonlat[index]
    print(lonlat)
    data = data[index]

    if isGrid:
        nlon, nlat = np.meshgrid(newlon, newlat)
        lonf = nlon.flatten()
        latf = nlat.flatten()
        nlonlat = np.array([lonf, latf]).T
    else:
        lonf=newlon
        latf=newlat
        nlonlat = np.array(list(zip(newlon,newlat)))

    if method == 'griddate':
        newdata = interpolate.griddata(lonlat, data, nlonlat)
        #newdata = pd.DataFrame(newdata).fillna(missingValue)
    elif method == 'idw':
        idw_tree = tree(lonlat, data)
        newdata = idw_tree(np.stack((lonf, latf), axis=1))
    newdata = np.round(newdata, 3)
    if isGrid:
        return newdata.reshape(len(newlat),len(newlon))
    else:
        return newdata

#Suitable for site interpolation, not for grid interpolation
def bilinear(data, lat, lon, info):
    '''
    using bilinear interpolation: estimate of unknown pixels(hydropower station)
    by their four nearest neighbours with proper weights(w)
    '''
    # Initial point and reolution of latitude and longitude
    '''
    parameter:
              info :dataframe ,columns=[newlat,newlon] 
    '''
    lat0 = lat[0]
    dlat = lat[1] - lat0
    lon0 = lon[0]
    dlon = lon[1] - lon0
    # left-lower nearest neighbour point and weights of their four nearest neighbours
    # loc=np.where((info.lat<np.min(lat))|(info.lat>np.max(lat))|(info.lon<np.min(lon))|(info.lon>np.max(lon)))[0]
    info.loc[
        (info.lat < np.min(lat)) | (info.lat > np.max(lat)) | (info.lon < np.min(lon)) | (info.lon > np.max(lon)), [
            'lat', 'lon']] = np.nan
    latz = np.modf((info.lat - lat0) / dlat)
    lonz = np.modf((info.lon - lon0) / dlon)
    latn = latz[1].values.astype('int')
    latn[latn < 0] = 0
    lonn = lonz[1].values.astype('int')
    lonn[lonn < 0] = 0
    latd = latz[0].values
    lond = lonz[0].values
    data = np.pad(data, ((0, 1), (0, 1)), 'constant', constant_values=(0, 0))
    ww = [(1 - latd) * (1 - lond), latd * (1 - lond), (1 - latd) * lond, latd * lond]
    # precipitation/temperature/cloudage above hydropower station
    newdata = data[latn, lonn] * ww[0] + data[latn + 1, lonn] * ww[1] + \
              data[latn, lonn + 1] * ww[2] + data[latn + 1, lonn + 1] * ww[3]
    #data = data[:len(lat), :len(lon)]
    #eg:(1000,)
    return newdata


def grid_interp_to_station(all_data, station_lon, station_lat, method='cubic'):
    '''
    func: 将等经纬度网格值 插值到 离散站点。使用griddata进行插值
    inputs:
        all_data,形式为：[grid_lon,grid_lat,data] 即[经度网格，纬度网格，数值网格]
        station_lon: 站点经度
        station_lat: 站点纬度。可以是 单个点，列表或者一维数组
        method: 插值方法,默认使用 cubic
    return:
        station_value: 插值得到的站点值
    '''
    station_lon = np.array(station_lon).reshape(-1, 1)
    station_lat = np.array(station_lat).reshape(-1, 1)

    lon = all_data[0].reshape(-1, 1)
    lat = all_data[1].reshape(-1, 1)
    data = all_data[2].reshape(-1, 1)

    points = np.concatenate([lon, lat], axis=1)

    station_value = interpolate.griddata(points, data, (station_lon, station_lat), method=method)

    station_value = station_value[:, :, 0]

    return station_value