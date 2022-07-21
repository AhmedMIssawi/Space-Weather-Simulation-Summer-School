# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:34:49 2022

@author: Ahmed Issawi
"""

import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt
import argparse

#r'C:\Users\Ahmed Issawi\Desktop\SpaceWeather- Simulation Summer School\Day2\omni_min_def_gOpl9_YZTo.lst'
def plot_symh(url):
    year=[]
    day=[]
    hour=[]
    minute=[]
    time = [] 
    SYMH=[]
    with open(r'{}'.format(url)) as f:
        for line in f:
            tmp = line.split()
            year.append(int(tmp[0]))
            day.append(int(tmp[1]))
            hour.append(int(tmp[2]))
            minute.append(int(tmp[3]))
            SYMH.append(int(tmp[4]))
            
            # create our date 
            time0 = dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0) + dt.timedelta(days=int(tmp[1])-1)
            time.append(time0)
            
            data = {'year':year,'day':day,'hour':hour,'minute':minute,'time':time,
                                'SYMH':SYMH}
    df = pd.DataFrame(data)

    time = df['time']
    data = df['SYMH']
    fig,ax = plt.subplots(figsize=(15, 15))
    ax.plot(time, data, marker='.', label='All Events')
    # apply < operator on array, then lp is apool numpy array 
    
    ax.axhline(df['SYMH'].max(), color="red", linestyle="--")
    ax.axhline(df['SYMH'].min(), color="red", linestyle="--")
    ax.axvline(x = df['time'][df['SYMH'].idxmax()], color = 'r',linestyle=":")
    ax.axvline(x = df['time'][df['SYMH'].idxmin()], color = 'r',linestyle=":")
    
    
    lp = data <-100
    # pass lp to subscript operator []
    #it returns a new array containing elements in the position of true in lp 
    
    ax.plot(time[lp],data[lp] ,marker='+', linestyle='',c='tab:orange',label='<-100 nT')
    ax.set_xlabel('year of 2013')
    ax.set_ylabel('SYMH (nT)')
    ax.grid(True)
    ax.legend()
    
    fig.savefig(f'{url}-100dpi.png', dpi = 100)
    print(df['SYMH'].min())
    print(df['time'][df['SYMH'].idxmax()].isoformat())
    
def parse_args():
  # Create an argument parser:
  parser = argparse.ArgumentParser(description = \
                                    'plot a graph between time and SYMH')
  # in_var: list of 2, no type defined -> string:
  parser.add_argument('url', \
                      help = 'my scalar variable',type = str)
      
  # actually parse the data now:
  args = parser.parse_args()
  return args



# parse the input arguments:
args = parse_args()
print(args)
# grab the variable in_var 
#   (note, this will be a list of 2 elements):
url = args.url
print(url)
if __name__ == '__main__':
    plot_symh(url)
                                
