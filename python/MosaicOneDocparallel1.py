from osgeo import gdal
import os
import glob
import math
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

#获取影像的左上角和右下角坐标
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

def GetData(in_filename):
    in_ds=gdal.Open(in_filename)
    in_gt=in_ds.GetGeoTransform()
    #仿射逆变换
    resul=[]
    offset=gdal.ApplyGeoTransform(inv_geotrans,in_gt[0],in_gt[3])
    x,y=map(int,offset)
    data=in_ds.ReadAsArray()
    resul.append(abs(x))
    resul.append(abs(y))
    resul.append(data)

    return resul

if __name__ == '__main__':
    star=time.time()

    path=r"/data/appdata/lai_param_TwoCycle_mask/2001"
    os.chdir(path)
    #如果存在同名影像则先删除
    # if os.path.exists('LaiParamOneCycle_2018.tif'):
    #     os.remove('LaiParamOneCycle_2018.tif')
        
    in_files=glob.glob("*.tif")#得到该目录下所有的影像名

    in_fn=in_files[0]
    #获取待镶嵌栅格的最大最小的坐标值
    min_x,max_y,max_x,min_y=GetExtent(in_fn)
    for in_fn in in_files[1:]:
        minx,maxy,maxx,miny=GetExtent(in_fn)
        min_x=min(min_x,minx)
        min_y=min(min_y,miny)
        max_x=max(max_x,maxx)
        max_y=max(max_y,maxy)
    # print(min_x,max_y,max_x,min_y)

    #计算镶嵌后影像的行列号
    in_ds=gdal.Open(in_files[0])
    geotrans=list(in_ds.GetGeoTransform())
    width=geotrans[1]
    height=geotrans[5]
    columns=math.ceil((max_x-min_x)/width)
    rows=math.ceil((max_y-min_y)/(-height))
    band_type=in_ds.GetRasterBand(1)
    in_band=in_ds.RasterCount
    #定义仿射逆变换
    inv_geotrans=gdal.InvGeoTransform(geotrans)

    pool = ThreadPoolExecutor(max_workers=72)
    result=list(pool.map(GetData,in_files))
    pool.shutdown(wait=True)

    # name="/data/appdata/lai_param_TwoCycle_mask/merge/LaiParamTwoCycle_2000.tif"
    name="/data/appdata/lai_param_TwoCycle_mask/merge/LaiParamTwoCycle_2001.tif"
    out_ds=gdal.GetDriverByName('GTiff').Create(name,columns,rows,in_band,band_type.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0]=min_x
    geotrans[3]=max_y
    out_ds.SetGeoTransform(geotrans)

    for j in range(len(in_files)):
        data=result[j]
        x=data[0]
        y=data[1]
        dataset=data[2]
        for i in range(in_band):
            out_ds.GetRasterBand(i+1).WriteArray(dataset[i,:,:],x,y)
    out_ds.FlushCache()  # 将数据写入硬盘
    out_ds = None
    
    
    print("total time:",time.time()-star)