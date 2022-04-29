from osgeo import gdal
import os
import glob
import math
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def GetExtent(in_fn):
    ds=gdal.Open(in_fn)
    geotrans=list(ds.GetGeoTransform())
    xsize=ds.RasterXSize
    ysize=ds.RasterYSize
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]
    ds=None
    return min_x,max_y,max_x,min_y

def GetTransform(in_files):
    min_x,max_y,max_x,min_y=GetExtent(in_files[0])
    for in_fn in in_files[1:]:
        minx,maxy,maxx,miny=GetExtent(in_fn)
        min_x=min(min_x,minx)
        max_y=max(max_y,maxy)
        max_x=max(max_x,maxx)
        min_y=min(min_y,miny)
    return min_x,max_y,max_x,min_y

def GetData(in_filename):
    # in_filename=dataset[0]
    # inv_geotrans=dataset[1]
    in_ds=gdal.Open(in_filename)
    geotrans=in_ds.GetGeoTransform()
    #仿射逆变换
    # inv_geotrans=gdal.InvGeoTransform(geotrans)
    offset=gdal.ApplyGeoTransform(inv_geotrans,geotrans[0],geotrans[3])
    x,y=map(int,offset)
    resul=[]
    # x=np.ceil(abs(geotrans[0]-min_x)/abs(geotrans[1]))
    # y=np.ceil(abs(geotrans[3]-max_y)/abs(geotrans[5]))

    data=in_ds.ReadAsArray()
    resul.append(abs(x))
    resul.append(abs(y))
    resul.append(data)

    return resul

def getTifInform(in_files):
    in_ds=gdal.Open(in_files[0])
    geotrans=in_ds.GetGeoTransform()
    Projection=in_ds.GetProjection()
    countbands=in_ds.RasterCount

    inv_geotrans=gdal.InvGeoTransform(geotrans)
    min_x,max_y,max_x,min_y=GetTransform(in_files)
    columns=np.ceil(abs(max_x-min_x)/abs(geotrans[1]))
    rows=np.ceil(abs(min_y-max_y)/abs(geotrans[5]))
    newgeotrans=list(geotrans)
    newgeotrans[0]=min_x
    newgeotrans[3]=max_y
    return int(columns),int(rows),countbands,Projection,newgeotrans,inv_geotrans

path=r"/data/appdata/lai_param_TwoCycle_mask/2007"

os.chdir(path)      
in_files=glob.glob("*.tif")#得到该目录下所有的影像名
min_x,max_y,max_x,min_y=GetTransform(in_files)
columns,rows,count,Projection,newgeotrans,inv_geotrans=getTifInform(in_files)

pool = ThreadPoolExecutor(max_workers=72)
result=list(pool.map(GetData,in_files))
pool.shutdown(wait=True)

name="/nvme1/LaiParamTwoCycle_2007"
out_ds=gdal.GetDriverByName('GTiff').Create(name,columns,rows,count,gdal.GDT_Float32)
out_ds.SetProjection(Projection)
out_ds.SetGeoTransform(newgeotrans)

for j in range(len(in_files)):
    data=result[j]
    x=data[0]
    y=data[1]
    dataset=data[2]
    # print(x,y)
    for i in range(count):
        out_ds.GetRasterBand(i+1).WriteArray(dataset[i,:,:],x,y)
out_ds.FlushCache()  # 将数据写入硬盘
out_ds = None