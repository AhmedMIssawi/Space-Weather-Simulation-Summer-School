# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:07:41 2022

@author: Ahmed Issawi
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
#import numpy as np 

#dataset= nc.Dataset('/c/Users/Ahmed Issawi/SWSSS2022/Space-Weather-Simulation-Summer-School/day_4/wfs.t12z.ipe05.20220720_090500.nc')
url = '/c/Users/Ahmed Issawi/SWSSS2022/Space-Weather-Simulation-Summer-School/day_4/wfs.t12z.ipe05.20220720_090500.nc'
def plot_tec(url, figsize=(10,6)):
    """"
    2-dimensional ionosphere outputs (TEC) at 5-min cadence (*ipe05*.nc)
    url: type string
    the full path of the file for windows 
    
    dateset: loading the file url using NETCDF$ library to laod the file
    
    the dataset have to many variables 
    dimensions(sizes): x01(90), x02(91), x03(109)
    variables(dimensions): float32 lon(x01), float32 lat(x02), float32 alt(x03), float32 tec(x02, x01)
    
    the output is the longitude with the latitude with colormesh Total electron contant 
    """
    dataset= nc.Dataset(f'{url}')
    fig, ax = plt.subplots(1,figsize=figsize)
    
    plt.pcolormesh(dataset['lon'][:],dataset['lat'][:],dataset['tec'][:])
    #pcolormesh y comes first then x 
    plt.colorbar(label=f'{dataset["tec"].units}')
    ax.set_title(f'WAM_IPE {dataset["tec"].units} {dataset.fcst_date}',fontsize=16)
    ax.set_ylabel('Longitude', fontsize=12)
    ax.set_xlabel('Latitude', fontsize=12)
    return fig, ax

plot_tec(url,figsize=(12,6))


#plt.pcolormesh(np.array(dataset['lat'][:]),np.array(dataset['lon'][:]),np.array(dataset['tec'][:]))
