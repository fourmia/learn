#!/usr/bin/env python
# coding=utf8
from . import DataBlock_pb2
import struct
import xarray as xr
import numpy as np
import datetime as dt
from .read_pup import ReadPup


def calc_scale_and_offset(min_v, max_v, n=16):
    # stretch/compress data to the avaiable packed range
    if max_v - min_v == 0:
        scale_factor = 1.0
        add_offset = 0.0
    else:
        scale_factor = (max_v - min_v) / (2**n -1)
        # translate the range to be symmetric about zero
        add_offset = min_v + 2 ** (n -1) * scale_factor
    return scale_factor, add_offset

def sv2nc(data,svname):
    ds = xr.Dataset()
    ds.coords['lon'] = ('lon', data.lon)
    ds['lon'].attrs['units'] = "degrees_east"
    ds['lon'].attrs['long_name'] = "Longitude"
    
    ds.coords['lat'] = ('lat', data.lat)
    ds['lat'].attrs['units'] = "degrees_north"
    ds['lat'].attrs['long_name'] = "Latitude"
    var = data.data
    scale_factor, add_offset = calc_scale_and_offset(np.min(var),
                                                     np.max(var))
    var = np.short((var - add_offset) / scale_factor)
    missingvalue = -999
    varname = 'Var'
    ds[varname] = (('lat', 'lon'), var)
    ds[varname].attrs['add_offset'] = add_offset
    ds[varname].attrs['scale_factor'] = scale_factor
    ds[varname].attrs['_FillValue'] = np.short((missingvalue - add_offset) / scale_factor)
    ds.to_netcdf(svname, format='NETCDF3_CLASSIC')
    ds.close()

class read:
#    tag: data type, 0 - Micaps, 1 - AWS, 2 - radar pup
    def __init__(self, response, tag = 0):
        ByteArrayResult = DataBlock_pb2.ByteArrayResult()
        ByteArrayResult.ParseFromString(response)
        if ByteArrayResult is not None:
            byteArray = ByteArrayResult.byteArray
            self.byteArray = byteArray
            if tag == 0:
                discriminator =struct.unpack("4s",byteArray[:4])[0].decode("gb2312")
                self.t = struct.unpack("h",byteArray[4:6])
                mName = struct.unpack("20s",byteArray[6:26])[0].decode("gb2312")
                eleName = struct.unpack("50s",byteArray[26:76])[0].decode("gb2312")
                description = struct.unpack("30s",byteArray[76:106])[0].decode("gb2312")
                self.level,self.y,self.m,self.d,self.h,self.timezone,self.period = struct.unpack("fiiiiii",byteArray[106:134])
                self.startLon,self.endLon,self.lonInterval,self.lonGridCount = struct.unpack("fffi",byteArray[134:150])
                self.startLat,self.endLat,self.latInterval,self.latGridCount = struct.unpack("fffi",byteArray[150:166])
                self.isolineStartValue,self.isolineEndValue,self.isolineInterval =struct.unpack("fff",byteArray[166:178])
                self.gridCount = self.lonGridCount*self.latGridCount
                self.description = mName.rstrip('\x00')+'_'+eleName.rstrip('\x00')+"_"+str(self.level)+'('+description.rstrip('\x00')+')'+":"+str(self.period)                    
                data=[] 
                if (self.gridCount == (len(byteArray)-278)/4):
                    for i in range(self.gridCount):
                        gridValue = struct.unpack("f",self.byteArray[278+i*4:282+i*4])[0]
                        data.append(gridValue)
                    self.data = np.array(data).reshape(1, self.latGridCount, self.lonGridCount)
                elif (self.gridCount == (len(byteArray)-278)/8):
                    for i in range(self.gridCount*2):
                        gridValue = struct.unpack("f",self.byteArray[278+i*4:282+i*4])[0]
                        data.append(gridValue)
                    self.data = np.array(data).reshape(2, self.latGridCount, self.lonGridCount)
                self.lat = np.linspace(self.startLat, self.startLat + self.latInterval*(self.latGridCount-1), self.latGridCount)
                self.lon = np.linspace(self.startLon, self.startLon + self.lonInterval*(self.lonGridCount-1), self.lonGridCount)
                self.time = dt.datetime(self.y, self.m, self.d, self.h)
            elif tag == 2:
                self.pup = ReadPup(self.byteArray)
    #保存MICAPS4类数据            
    def sv2micaps(self, filename = None):
        if filename == None:
            filename = self.time.strftime("%Y%m%d%H") + ".%03d"%self.period
        with open (filename,'w') as writer:
            eachline = "diamond 4 "+self.description
            writer.write(eachline+"\n")
            eachline = str(self.y)+"\t"+str(self.m)+"\t"+str(self.d)+"\t"+str(self.h)+"\t"+str(self.period)+"\t"+str(self.level)+"\t"\
            +str(self.lonInterval)+"\t"+str(self.latInterval)+"\t"+str(round(self.startLon,2))+"\t"\
            +str(self.endLon)+"\t"+str(round(self.startLat,2))+"\t"+str(round(self.endLat,2))+\
            "\t"+str(self.lonGridCount)+"\t"+str(self.latGridCount)+"\t"+\
            str(self.isolineInterval)+"\t"+str(self.isolineStartValue)+"\t"+\
            str(self.isolineEndValue)+"    3    0"
            writer.write(eachline+"\n")
            for i in range(np.size(self.data,1)):
                for j in range(np.size(self.data,2)):
                    writer.write(str(round(self.data[0,i,j],2)).ljust(10))
                writer.write('\n')
    
    #保存NetCDF数据            
    def sv2nc(self, filename = None):
        if filename == None:
            filename = self.time.strftime("%Y%m%d%H") + ".%03d.nc"%self.period
        ds = xr.Dataset()
        ds.coords['lon'] = ('lon', self.lon)
        ds['lon'].attrs['units'] = "degrees_east"
        ds['lon'].attrs['long_name'] = "Longitude"
        
        ds.coords['lat'] = ('lat', self.lat)
        ds['lat'].attrs['units'] = "degrees_north"
        ds['lat'].attrs['long_name'] = "Latitude"
        
        ds['time'] = ('time', np.array([self.period]))
        ds['time'].attrs['units'] = self.time.strftime("hours since %Y-%m-%d %H:%M:%S")
        ds['time'].attrs['long_name'] = "Time(CST)"
        
        var = self.data
        scale_factor, add_offset = calc_scale_and_offset(np.nanmin(var),
                                                         np.nanmax(var))
        var = np.short((var - add_offset) / scale_factor)
        missingvalue = -999
        varname = 'Var'
        ds[varname] = (('time', 'lat', 'lon'), var)
        ds[varname].attrs['add_offset'] = add_offset
        ds[varname].attrs['scale_factor'] = scale_factor
        ds[varname].attrs['_FillValue'] = np.short((missingvalue - add_offset) / scale_factor)
        ds.to_netcdf(filename, format='NETCDF3_CLASSIC')
        ds.close()
