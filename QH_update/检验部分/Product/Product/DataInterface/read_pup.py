# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 09:20:10 2017
@version 2
@author: LXL
#ReadPup类：读取雷达产品数据pup
#convert_coord函数：转变直角坐标为经纬度
"""

from struct import unpack
import numpy as np
from scipy.interpolate import griddata
import pandas as pd


#%%
class ReadPup:
    def __init__(self, byteArray):
        self.stream = byteArray
        self.data = unpack('%sB' % len(self.stream), self.stream)
        self.read_data()
        
    #==========================================================================
    # 读取一个单字节或双字节    
    # double参数代表读双字节，divided参数代表将一个字节分成4+4bit的距离颜色对
    def get_data(self, p, double=False, divided=False):
        #按单字节、双字节和半个字节读
        if double:
            v = self.data[2*p-2]*256 + self.data[2*p-1]
        elif divided:
            v = (self.data[p]//16, self.data[p]%16)
        else:
            v = self.data[p]
        return v
    
    def read_data(self):
        '''头文件,信息描述'''
        # ========产品代号及雷达经纬度========
        self.code = self.get_data(1,True)
        self.Lat0 = (self.get_data(11,True)*65536+self.get_data(12,True))/1000.
        self.Lon0 = (self.get_data(13,True)*65536+self.get_data(14,True))/1000.
        self.alt0 = self.get_data(15,True) * 0.3048
        
        # ========产品号===================
        # 根据产品号设置分辨率
        self.ProductCode = self.get_data(16,True)
        if self.ProductCode in (22,25,28):
            self.res = 0.25
        elif self.ProductCode in (23,26,29):
            self.res = 0.5
        elif self.ProductCode in (16,19,24,27,30,33,35,37,56,87,110):
            self.res = 1.0
        elif self.ProductCode in (17,20,78,79,80):
            self.res = 2.0
        elif self.ProductCode in (18,21,36,38,41,57):
            self.res = 4.0
        else:
            self.res = 1.0
        #======体扫模式和仰角===============
        #仰角是角度制
        self.Mode = self.get_data(18,True)
        if self.Mode == 21:
            B = [0, 0.5, 1.5, 2.4, 3.4, 4.3, 6.0, 9.9, 14.6, 19.5]
        elif self.Mode == 11:
            B = [0, 0.5, 1.5, 2.4, 3.4, 4.3, 5.3, 6.2, 7.5, 8.7, 10.0, 12.0, 14.0, 16.7, 19.5]
        else:
            B = [0, 0.5, 1.5, 2.5, 3.5, 4.5]
        self.Ecode = self.get_data(29,True)#体扫模式
        self.Elev = B[self.Ecode+1]
        #======日期和时间==================
        # 数值为1时表示1970年1月1日,7.	时间（当天的第几秒）
        self.Date = self.get_data(21,True)
        self.Time = self.get_data(22,True)*65536+self.get_data(23,True)
        #======产品数据块==================
        # 大于0表示存在
        # 产品符号表示块
        """栅格数据包符号块代码高低位次与径向数据包的不一样，为什么？"""
        if self.ProductCode in (35,36,37,38,39,40,41,42,47,53,57,58,62,63,64,65,66,89,90): 
            self.PSB = self.get_data(55,True)*65536+self.get_data(56,True)
        else:
            self.PSB = self.get_data(55,True)+self.get_data(56,True)*65536
        if self.PSB > 60:
            self.PSB = 60
        # 图像数字文本块        
        self.GAB = self.get_data(57,True)*65536+self.get_data(58,True)
        # 文本列表块
        self.TAB = self.get_data(59,True)*65536+self.get_data(60,True)

#######========================================================================        
        '''数据块部分'''
        if self.PSB > 0:
            p = self.PSB
            self.NLayer = self.get_data(65,True)
            self.sign = self.get_data(69,True)
            p += 5
            for layer in range(self.NLayer):
                LLayer = self.get_data(p+3,True)
                if LLayer == 0:
                    LLayer = self.get_data(p+2,True)
                else:
                    LLayer += self.get_data(p+2,True) * 65536
                p += 3
                #==============================================================
                # 径向数据包
                if self.sign == 44831:
                    # 第一个距离库的位置
                    self.FirstRangeBin = self.get_data(70,True)
                    #组成一条径向数据的库数
                    self.NumberOfRangeBins = self.get_data(71,True)
                    #中心点坐标I
                    self.ICOS = self.get_data(72,True)
                    #中心点坐标J
                    self.JCOS = self.get_data(73,True)
                    #比例因子
                    self.SFactor = self.get_data(74,True)/1000.
                    #径向数据条数
                    self.NumberOfRadials = self.get_data(75,True)
                    
                    self.Color = np.zeros((self.NumberOfRadials,self.NumberOfRangeBins))
                    self.X = np.zeros((self.NumberOfRadials,self.NumberOfRangeBins))
                    self.Y = np.zeros((self.NumberOfRadials,self.NumberOfRangeBins))
                    #指针
                    p = 150
                    for i in range(self.NumberOfRadials):
                        self.NumberOfRLE = self.get_data(p//2+1,True)
                        self.RSA = self.get_data(p//2+2,True)/10.
                        self.RAD = self.get_data(p//2+3,True)/10.
                        theta = self.RSA+self.RAD/2.
                        theta = theta*np.pi/180.
                        p = p+6
                        temp = 0
                        for j in range(2*self.NumberOfRLE):
                            r,c = self.get_data(p,divided=True)
                            rr = temp + r
                            self.Color[i,temp:rr] = c                                                        
                            self.X[i,temp:rr] = rr * np.sin(theta)*np.cos(self.Elev*np.pi/180)*self.res
                            self.Y[i,temp:rr] = rr * np.cos(theta)*np.cos(self.Elev*np.pi/180)*self.res
                            temp = rr
                            #移动指针
                            p = p+1                        
                        
                #==============================================================
                # 栅格数据包
                elif self.sign in (47631,47623):
    
                    self.IStart = self.get_data(72,True)
                    self.JStart = self.get_data(73,True)
                    self.XScale = self.get_data(74,True)
                    self.YScale = self.get_data(76,True)
                    self.NumberOfRows = self.get_data(78,True)                
                    self.Color = np.zeros((self.NumberOfRows,self.NumberOfRows))
                 
                    p = 158
                    for i in range(self.NumberOfRows):
                        self.NumberOfBytes = self.get_data(p//2+1,True)
                        p = p+2
                        s=0
                        for j in range(self.NumberOfBytes):
                            x,c = self.get_data(p,divided=True)
                            self.Color[i,s:s+x] = [c]*x
                            s=s+x                      
                            p = p+1
                    self.IStart = (self.IStart-(self.NumberOfRows-1))//2
                    self.JStart = (self.JStart-(self.NumberOfRows-1))//2
                    self.X = np.arange(self.IStart,self.IStart+self.NumberOfRows)*self.res
                    self.Y = np.arange(self.JStart,self.JStart+self.NumberOfRows)*self.res          
                    
                #==============================================================
                # 向量数据包
                else:
                    pend = p + int(LLayer/2)
                    # 用列表存储数据
                    self.Xc = []
                    self.Yc = []
                    self.Jc = []
                    self.Wc = []
                    self.rc = []
                    self.text_data = []
                    self.Xa = []
                    self.Ya = []
                    while(p<pend):
                        self.sign = self.get_data(p+1,True)
                        p += 1
                        if self.sign in (6,7,9,10):
                            
                            Pagelen = self.get_data(p+1,True)
                            p += 1
                            
                            if self.sign in (6,9):
                                if self.sign == 9:
                                    p += 1
                                    Pagelen -= 2
                                Xa = []
                                Ya = []
                                temp = self.get_data(p+1,True)
                                if temp > 10000:
                                    temp -=65536
                                Xa.append(temp/4.0)
                                temp = self.get_data(p+2,True)
                                if temp > 10000:
                                    temp -=65536
                                Ya.append(temp/4.0)
                                for kk in range((Pagelen - 4)/4):
                                    temp = self.get_data(p+3+kk*2,True)
                                    if temp > 10000:
                                        temp -=65536
                                    Xa.append(temp/4.0)
                                    temp = self.get_data(p+4+kk*2,True)
                                    if temp > 10000:
                                        temp -=65536
                                    Ya.append(temp/4.0)
                                self.Xa.append(Xa)
                                self.Ya.append(Ya)
                                    
                            if self.sign in (7,10):
                                if self.sign == 10:
                                    p += 1
                                    Pagelen -= 2
                                Xa = []
                                Ya = []
                                for kk in range((Pagelen)/4):
                                    temp = self.get_data(p+3+kk*2,True)
                                    if temp > 10000:
                                        temp -=65536
                                    Xa.append(temp/4.0)
                                    temp = self.get_data(p+4+kk*2,True)
                                    if temp > 10000:
                                        temp -=65536
                                    Ya.append(temp/4.0)
                                self.Xa.append(Xa)
                                self.Ya.append(Ya)
                            # 指针
                            p += int(Pagelen/2)
                            
                                    
                        #==============================================================
                        # 文本数据包
                        # 
                        elif self.sign in (1,2,8,11,15):
        
                            #p=self.PSB
                            #页长度
                            Pagelen = self.get_data(p+1,True)
                            p += 1
                            
                            if self.sign == 8:
                                p += 1
                                Pagelen -= 2
                                
                            tempx = self.get_data(p+1,True)
                            if tempx > 10000:
                                tempx -=65536
                            tempx /= 4.0
                            self.Xc.append(tempx)
                            tempy = self.get_data(p+2,True)
                            if tempy > 10000:
                                tempy -=65536
                            tempy /= 4.0
                            self.Yc.append(tempy)
                            [Jc,Wc] = conv_coord(tempx,tempy,self.Lon0,self.Lat0)
                            self.Jc.append(round(Jc,2))
                            self.Wc.append(round(Wc,2))
                            td = ''
                            for kk in range(Pagelen - 4):
                                td = td + chr(self.get_data(2*p+4+kk))
                            self.text_data.append(td)
                            # 指针
                            p += int(Pagelen/2)
                            '''
                        elif self.sign == 3:
                            pagelen = self.get_data(p+1,True)
                            self.rc.append(self.get_data(p+4,True))
                            p = p + pagelen/2 + 1
                            '''
                            
                        else:
                            pagelen = self.get_data(p+1,True)
                            p = p + int(pagelen/2) + 1

                        
#######========================================================================
#         图像数字文本块
        if (self.GAB > 0) and (self.ProductCode == 38):
            p=self.GAB
            # 页数
            #
            self.NPages = self.get_data(p+5,True)
            # 用列表存储数据
#            print self.NPages
            self.gab_data = []
            p = p + 5
            for i in range(self.NPages):
                #页长度
                
                self.PageNum = self.get_data(p+1,True)
                
                self.Pagelen = self.get_data(p+2,True)                
                p = p + 2    
                p0 = p
                while p < (p0+self.Pagelen/2):
                    #块标识
#                    print p,p0+self.Pagelen/2
                    self.sign_gab = self.get_data(p+1,True)
                    """
                    #只有一层时Length应该接近于Pagelen的，但事实不是
                    """
                    self.Length = self.get_data(p+2,True)
#                    print self.sign_gab
                    if self.sign_gab == 8:
                        self.N = self.Length - 6
                        self.ColorLevel = self.get_data(p+3,True)
                        self.I = self.get_data(p+4,True)
                        self.J = self.get_data(p+5,True)
                        self.gab_data.append(self.stream[2*p+10:2*p+10+self.N])
                        p = p+5+int((self.N)/2)
                        
                    if self.sign_gab in (1,2,15):
                        self.N = self.Length - 4
                        self.I = self.get_data(p+3,True)
                        self.J = self.get_data(p+4,True)
                        self.gab_data.append(self.stream[2*p+8:2*p+8+self.N])
                        p = p+4+(self.N)/2
                        
                    if self.sign_gab in (7,10):
                        self.N = self.Length - 4
                        self.I = self.get_data(p+3,True)
                        self.J = self.get_data(p+4,True)
                        p = p+4+int((self.N)/2)
                        
            self.hail = hail_to_pd(self.NPages,self.gab_data,self.Elev,self.res,self.Lon0,self.Lat0)


#%%
def conv_coord(X,Y,Lon0,Lat0):
    R = 6371.
    J = np.arcsin((4*R*X)/(np.square(X) + np.square(Y)+4*R*R)/np.cos(Lat0*np.pi/180))
    J = J*180/np.pi + Lon0
    A = np.arcsin((4*R*Y)/(np.square(X) + np.square(Y)+4*R*R))
    B = np.arctan(np.tan(Lat0*np.pi/180) * np.cos(J*np.pi/180 - Lon0*np.pi/180))
    W = (A + B)*180/np.pi
    return J,W
    
#%%
def hail_to_pd(num_page, str_list, Elev, res, Lon0, Lat0):
    l = int(len(str_list)/num_page)
    ID = []
    AZ = []
    RAN = []
    POSH = []
    POH = []
    MAX_ICESIZE = []
#    (ID, AZ, RAN, POSH, POH, MAX_ICESIZE) = [[]]*6
    for i in range(num_page):        
        for layer in str_list[i*l+1:i*l+l]:
            temp = layer[4:6] if not layer[4:6].isspace() else np.nan
            ID.append(layer[4:6])
            temp = float(layer[7:11]) if not layer[7:11].isspace() else np.nan
            AZ.append(temp)
            temp = float(layer[12:15]) if not layer[12:15].isspace() else np.nan
            RAN.append(temp)
            if layer[26:42].strip() != 'UNKNOWN':
                temp = float(layer[26:30]) if not layer[26:30].isspace() else np.nan
                POSH.append(temp)
                temp = float(layer[31:34]) if not layer[31:34].isspace() else np.nan
                POH.append(temp)
                temp = float(layer[36:42]) if not layer[36:42].isspace() else np.nan
                MAX_ICESIZE.append(temp)
            else:
                POSH.append(np.nan)
                POH.append(np.nan)
                MAX_ICESIZE.append(np.nan)
    
    f = pd.DataFrame({'ID':ID, 'AZ':AZ, 'RAN':RAN, 'POSH':POSH, 'POH':POH, 'MAX_ICESIZE':MAX_ICESIZE},
                     columns=['ID', 'AZ', 'RAN', 'POSH', 'POH', 'MAX_ICESIZE'])
    
    f['hail_risk'] = np.nan
    f.loc[(30<=f.POH) & (f.POH<50),'hail_risk'] = '1'
    f.loc[(50<=f.POH) & (f.POSH<30),'hail_risk'] = '2'
    f.loc[(30<=f.POSH) & (f.POSH<50),'hail_risk'] = '3'
    f.loc[f.POSH>=50,'hail_risk'] = '4'
#    f['hail_risk'][(30<=f.POH) & (f.POH<50)] = '1'
#    f['hail_risk'][(50<=f.POH) & (f.POSH<30)] = '2'
#    f['hail_risk'][(30<=f.POSH) & (f.POSH<50)] = '3'
#    f['hail_risk'][50<=f.POSH] = '4'
    
    """根据方位角和径向距离计算经纬度"""
    X = f.RAN * np.sin(f.AZ*np.pi/180)*np.cos(Elev*np.pi/180)#*res
    Y = f.RAN * np.cos(f.AZ*np.pi/180)*np.cos(Elev*np.pi/180)#*res
    
    J, W = conv_coord(X, Y, Lon0, Lat0)
    
    f['lon'] = J
    f['lat'] = W

    result = f[['lon','lat','hail_risk','ID']]
    """筛选""" 
    result = result.dropna(how='any')
    return result

#%%
def convert_coord(obj):
    """
    params:
        obj - object of read_pup class
    
    增加JWColor表示经纬度下的Color, add 'J' for longtitude and 'W' for latitude.
    """
    #======================================================
    #中心点坐标作为雷达坐标
    R = 6371.
    J = np.arcsin((4*R*obj.X)/(np.square(obj.X) + np.square(obj.Y)+4*R*R)/np.cos(obj.Lat0*np.pi/180))
    obj.J = J*180/np.pi + obj.Lon0
    A = np.arcsin((4*R*obj.Y)/(np.square(obj.X) + np.square(obj.Y)+4*R*R))
    B = np.arctan(np.tan(obj.Lat0*np.pi/180) * np.cos(obj.J*np.pi/180 - obj.Lon0*np.pi/180))
    obj.W = (A + B)*180/np.pi
    obj.JWColor = obj.Color
    #径向数据包还需要插值成直角坐标
    if obj.sign == 44831:
        X = np.vstack((obj.J.flatten(),obj.W.flatten())).T
        z = obj.Color.flatten()
        newlon = np.arange(np.min(obj.J), np.max(obj.J), obj.res/100.)
        newlat = np.arange(np.min(obj.W), np.max(obj.W), obj.res/100.)
        newX = np.meshgrid(newlon,newlat)
        grid_shape = newX[0].shape
        newX = np.reshape(newX, (2,-1)).T

        newdata = griddata(X,z,newX,method='linear').reshape(grid_shape)
        newdata = np.nan_to_num(newdata)
        newdata = newdata[::-1]
        obj.JWColor = newdata
        obj.J = newlon
        obj.W = newlat

