# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:00:01 2022

@author: Ahmed Issawi, Kaitlin Doublestein, Brians Chinonso Amadi, Neha Srivastava
__email__=

"""
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd

#%% loading data JB and TIE-GSM
"""
TASK 1.
"""

dir_density_Jb2008 = r'C:\Users\Ahmed Issawi\SWSSS2022\Space-Weather-Simulation-Summer-School\day_5\project 1\JB2008\2002_JB2008_density.mat'
# Load Density Data
try:
    loaded_data_JB = loadmat(dir_density_Jb2008)
except:
    print("File not found. Please make sure you have input the correct directory")

import h5py

loaded_data_TI= h5py.File(r'C:\Users\Ahmed Issawi\SWSSS2022\Space-Weather-Simulation-Summer-School\day_5\project 1\TIEGCM\2002_TIEGCM_density.mat')
#%% loading Omni data 

def read_ascii_file(file):
    """
    Args:
        file(str):
            Name of the file that is to be read. Designed to read data from 
            OMNI database.
        
    Return:
        dict_data(dictionary):
            dictionary of the time for the data. The data used follows the
            format YYYY-DD HH:MM. Does not contain month.
        
    Example:
        data_dict = read_ascii_file(:C\[PATH]\{filename})

    """
    with open(file) as f:
        year = []
        day = []
        hour = []
        minute = []
        time = []
        data = []
        for line in f:
            tmp = line.split()
            year.append(int(tmp[0]))
            day.append(int(tmp[1]))
            hour.append(int(tmp[2]))
            minute.append(int(tmp[3]))
            data.append(float(tmp[4]))

            #create datetime
            time0 = dt.datetime(int(tmp[0]),1,1,int(tmp[2]),int(tmp[3]),0)\
                    + dt.timedelta(days=int(tmp[1])-1)
            time.append(time0)
            
        dict_data = {'time': time,
                     'data': data
            }
    return dict_data
#change the Ascii data to pandas data frame
df_dict =read_ascii_file(r'C:\Users\Ahmed Issawi\SWSSS2022\Space-Weather-Simulation-Summer-School\day_5\project 1\omni_min_def_sLntSlJiFv.lst')
df = pd.DataFrame(df_dict) # df meaning data frame 

#%% plot the area of interest from omni data 

"""
Slices the data into a point of interest. Time selected was by plotting the
time and data to identify important high dst times.
"""
data_sliced=df.iloc[df[df.time == '2002-04-20 01:00:00'].index[0]:df[df.time == '2002-04-20 11:00:00'].index[0]]

fig, ax = plt.subplots(1,figsize=(10,5))
plt.plot(data_sliced['time'],data_sliced['data'],color='r')
ax.set_xlabel('Time period',fontsize=12)
ax.set_ylabel('SYM-H (nT)')
ax.set_title('SYM-H data plotted during 20-04-2002')

# shaded the part between the maximum and miniimum point
ax.axvspan(data_sliced['time'][data_sliced['data']==data_sliced['data'].max()], \
           data_sliced['time'][data_sliced['data']==data_sliced['data'].min()], color='k', alpha=0.4, lw=0)

#change_xtick labels and rotation
hourslice = list(data_sliced['time'][0::30].astype(str)) # convert the time slice to string for each hour
hour = [hour[-8:] for hour in hourslice] # slice the strings of the date after converting 

plt.xticks(data_sliced['time'][0::30],list(hour),rotation = 60) # rotation is 60 degree and the x tick set for every hour 
ax.grid(True)
plt.savefig("Omnidata.png",dpi=100)
plt.show()

#save plot



#%% variables for the JB and TIE-GSM data 
"""

TASK 2.
Extract and plot the predicted at 450 km the selected period of high space 
activity.
"""
# data variables 

JB2008_dens = loaded_data_JB['densityData']

localSolarTimes_JB2008 = np.linspace(0,24,24)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0]
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
nofLat_JB2008 = latitudes_JB2008.shape[0]

time_array_JB2008 = np.linspace(0,8760,20, dtype = int)

tiegcm_dens = (10**np.array(loaded_data_TI['density'])*1000).T # convert from g/cm3 to kg/m3
altitudes_tiegcm = np.array(loaded_data_TI['altitudes']).flatten()
latitudes_tiegcm = np.array(loaded_data_TI['latitudes']).flatten()
localSolarTimes_tiegcm = np.array(loaded_data_TI['localSolarTimes']).flatten()
nofAlt_tiegcm = altitudes_tiegcm.shape[0]
nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
nofLat_tiegcm = latitudes_tiegcm.shape[0]

time_array_tiegcm = np.linspace(0,8760,20, dtype = int)
#%% data reshape fot JB and TIE-GSM data

JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008,8760), order='F')

tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')

#%% comparison data plots between JB and TIE-GSM
from scipy.interpolate import RegularGridInterpolator

"""
Interpolating JB2008 and TIE-GCM with the density data

Plotting the interpolated data in a contour plot.
"""

time =  list(range(2635,2645,1))

for time_index in time:
    #inter function is the function that will interpolate over the data 
    inter_function = RegularGridInterpolator((localSolarTimes_tiegcm, latitudes_tiegcm, altitudes_tiegcm), tiegcm_dens_reshaped[:,:,:,time_index], bounds_error=False, fill_value=None)


    tiegcm_jb2008_grid = np.zeros((24,20))
    for lst_i in range (24):
        for lat_i in range (20):
            tiegcm_jb2008_grid[lst_i,lat_i] = inter_function((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],450))

    fig, axs = plt.subplots(2, figsize=(15, 10), sharex=True)

    cs = axs[0].contourf(localSolarTimes_JB2008, latitudes_JB2008, tiegcm_jb2008_grid.T, cmap='jet')
    
    ax.clabel(cs, inline=True,colors='k', fontsize=14)#draw the difference values on the graphs
    
    axs[0].set_title('TIE-GCM density at 450 km, t = {} hrs'.format(time_index), fontsize=18)
    axs[0].set_ylabel("Latitudes", fontsize=18)
    axs[0].tick_params(axis = 'both', which = 'major', labelsize = 16)
        
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs,ax=axs[0])
    cbar.ax.set_ylabel('Density')
    
    #inter function is the function that will interpolate over the data
    inter_function = RegularGridInterpolator((localSolarTimes_JB2008, latitudes_JB2008, altitudes_JB2008), JB2008_dens_reshaped[:,:,:,time_index], bounds_error=False, fill_value=None)

    jb2008_tiegcm_grid = np.zeros((24,20))
    for lst_i in range (24):
        for lat_i in range (20):
            jb2008_tiegcm_grid[lst_i,lat_i] = inter_function((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],450))

    alt = 450

    cs = axs[1].contourf(localSolarTimes_JB2008, latitudes_JB2008, jb2008_tiegcm_grid.T, cmap='jet')
    
    ax.clabel(cs, inline=True,colors='k', fontsize=14)#draw the difference values on the graphs
    
    axs[1].set_title('JB2008 density at 450 km, t = {} hrs'.format(time_index), fontsize=18)
    axs[1].set_ylabel("Latitudes", fontsize=18)
    axs[1].tick_params(axis = 'both', which = 'major', labelsize = 16)
        
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs,ax=axs[1])
    cbar.ax.set_ylabel('Density')

    axs[1].set_xlabel("Local Solar Time", fontsize=18)

    #save data    
    plt.savefig(f"comparison{time_index}.png",dpi=100)
#%% Absolute percentage difference of the JB and TIE-GSM data
"""

TASK 3.

Plotting the absolute percent difference.

"""
for time_index in time:
    #inter functions of TI and JB is the function that will interpolate over the data for the both data set 
    inter_function_ti = RegularGridInterpolator((localSolarTimes_tiegcm, latitudes_tiegcm, altitudes_tiegcm), tiegcm_dens_reshaped[:,:,:,time_index], bounds_error=False, fill_value=None)
    inter_function_jb = RegularGridInterpolator((localSolarTimes_JB2008, latitudes_JB2008, altitudes_JB2008), JB2008_dens_reshaped[:,:,:,time_index], bounds_error=False, fill_value=None)


    tiegcm_jb2008_grid = np.zeros((24,20))
    for lst_i in range (24):
        for lat_i in range (20):
            tiegcm_jb2008_grid[lst_i,lat_i] = inter_function_ti((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],450))



    jb2008_tiegcm_grid = np.zeros((24,20))
    for lst_i in range (24):
        for lat_i in range (20):
            jb2008_tiegcm_grid[lst_i,lat_i] = inter_function_jb((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],450))
            
            
    abs_percent_diff = abs(jb2008_tiegcm_grid - tiegcm_jb2008_grid)/tiegcm_jb2008_grid
    
    print(abs_percent_diff.max())
    
    fig, axs = plt.subplots(1, figsize=(15, 10))

    cs = axs.contourf(localSolarTimes_JB2008, latitudes_JB2008, abs_percent_diff.T, cmap='jet')
    
    ax.clabel(cs, inline=True,colors='k', fontsize=14)#draw the difference values on the graphs
    
    
    axs.set_title('Absolute percentage difference between TIE-GCM and JB2008 at 450 km, t = {} hrs'.format(time_index), fontsize=18)
    axs.set_ylabel("Latitudes", fontsize=18)
    axs.set_xlabel("Solar local time", fontsize=18)
    axs.tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs,ax=axs)
    cbar.ax.set_ylabel('Absolute percentage difference')
    
    #save data
    plt.savefig(f"Absolute percentage{time_index}.png",dpi=100)


#%% Density difference between JB and TIE-GSM data 

for time_index in time:
    #inter functions of TI and JB is the function that will interpolate over the data for the both data set
    inter_function_ti = RegularGridInterpolator((localSolarTimes_tiegcm, latitudes_tiegcm, altitudes_tiegcm), tiegcm_dens_reshaped[:,:,:,time_index], bounds_error=False, fill_value=None)
    inter_function_jb = RegularGridInterpolator((localSolarTimes_JB2008, latitudes_JB2008, altitudes_JB2008), JB2008_dens_reshaped[:,:,:,time_index], bounds_error=False, fill_value=None)


    tiegcm_jb2008_grid = np.zeros((24,20))
    for lst_i in range (24):
        for lat_i in range (20):
            tiegcm_jb2008_grid[lst_i,lat_i] = inter_function_ti((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],450))



    jb2008_tiegcm_grid = np.zeros((24,20))
    for lst_i in range (24):
        for lat_i in range (20):
            jb2008_tiegcm_grid[lst_i,lat_i] = inter_function_jb((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],450))
            
            
    diff = jb2008_tiegcm_grid - tiegcm_jb2008_grid
    
    print(abs_percent_diff.max())
    
    fig, axs = plt.subplots(1, figsize=(15, 10))

    cs = axs.contourf(localSolarTimes_JB2008, latitudes_JB2008, diff.T,cmap='jet') #
    
    
    ax.clabel(cs, inline=True,colors='k', fontsize=14)#draw the difference values on the graphs
    
    
    axs.set_title('Density difference between TIE-GCM and JB2008  at 450 km, t = {} hrs'.format(time_index), fontsize=18)
    axs.set_ylabel("Latitudes", fontsize=18)
    axs.set_xlabel("Solar local time", fontsize=18)
    axs.tick_params(axis = 'both', which = 'major', labelsize = 16)
    
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(cs,ax=axs)
    cbar.ax.set_ylabel('Density difference')
    
    #save data 
    plt.savefig(f"Density difference{time_index}.png",dpi=100)
#%%




