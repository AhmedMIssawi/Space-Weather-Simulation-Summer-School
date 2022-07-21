# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:56:30 2022

@author: Ahmed Issawi
"""


import netCDF4 as nc
import matplotlib.pyplot as plt
import argparse





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

#plot_tec(url,figsize=(12,6))

#plt.savefig(f'{dataset["tec"].units} {dataset.fcst_date}-100dpi.png', dpi = 100)


def parse_args():
  # Create an argument parser:
  parser = argparse.ArgumentParser(description = \
                                    'plot a pcolormesh between lon and lat with the TEC')
  # in_var: list of 2, no type defined -> string:
  parser.add_argument('urls', nargs='+', \
                      help = 'my scalar variable',type = str)
      
  # actually parse the data now:
  args = parser.parse_args()
  return args



# parse the input arguments:
args = parse_args()
print(args)
# grab the variable in_var 
#   (note, this will be a list of 2 elements):

urls = args.urls

for url in urls:
    out = url + '.png'
    print(out)
    plt.savefig(out, dpi = 100)
    if __name__ == '__main__':
        plot_tec(url)