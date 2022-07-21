# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:42:54 2022

@author: Ahmed Issawi
"""
# Import required packages
# loading dependences
import numpy as np
import h5py
import matplotlib.pyplot as plt 
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat


# not completed 

def plotter_v2(url1,url2,time,alt):
#%% load data section 

        dir_density_Jb2008 = 'JB2008/2002_JB2008_density.mat'
        # Load Density Data
        try:
            loaded_data = loadmat(dir_density_Jb2008)
            print (loaded_data)
        except:
            print("File not found. Please check your directory")
        
        # Uses key to extract our data of interest
        JB2008_dens = loaded_data['densityData']
        
        loaded_data = h5py.File('TIEGCM/2002_TIEGCM_density.mat')
        
        
        
        
        
        #%% variable section
        localSolarTimes_JB2008 = np.linspace(0,24,24)
        latitudes_JB2008 = np.linspace(-87.5,87.5,20)
        altitudes_JB2008 = np.linspace(100,800,36)
        nofAlt_JB2008 = altitudes_JB2008.shape[0]
        nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
        nofLat_JB2008 = latitudes_JB2008.shape[0]
        
        # For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
        JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008,8760), order='F')
        
        
        tiegcm_dens = (10**np.array(loaded_data['density'])*1000).T # convert from g/cm3 to kg/m3
        altitudes_tiegcm = np.array(loaded_data['altitudes']).flatten()
        latitudes_tiegcm = np.array(loaded_data['latitudes']).flatten()
        localSolarTimes_tiegcm = np.array(loaded_data['localSolarTimes']).flatten()
        nofAlt_tiegcm = altitudes_tiegcm.shape[0]
        nofLst_tiegcm = localSolarTimes_tiegcm.shape[0]
        nofLat_tiegcm = latitudes_tiegcm.shape[0]
    
        tiegcm_dens_reshaped = np.reshape(tiegcm_dens,(nofLst_tiegcm,nofLat_tiegcm,nofAlt_tiegcm,8760), order='F')
        
        #%% plots
        time_index=31*24
        #interpolation function 
        tiegcm_interpolated_fun = RegularGridInterpolator((localSolarTimes_tiegcm, latitudes_tiegcm, altitudes_tiegcm),\
                                                          tiegcm_dens_reshaped[:,:,:,time_index], bounds_error=False,fill_value=None)
        tiegcm_jb2008_grid = np.zeros((24,20))
        for lst_i in range(24):
            for lat_i in range(20):
                tiegcm_jb2008_grid[lst_i,lat_i]=tiegcm_interpolated_fun((localSolarTimes_JB2008[lst_i],latitudes_JB2008[lat_i],400))
        
        fig, axs = plt.subplots(2,figsize=(15,10),sharex=True)
        cs = axs[0].contourf(localSolarTimes_JB2008,latitudes_JB2008,tiegcm_jb2008_grid.T)
        axs[0].set_title('TIE-GCM density at 400 KM, t={} hrs'.format(time_index),fontsize=18)
        axs[0].set_ylabel('Latitudes', fontsize=18)
        axs[0].tick_params(axis='both',which='major',labelsize=16)
        cbar= fig.colorbar(cs,ax=axs[0])
        cbar.ax.set_ylabel('Density')
        alt = 400 
        hi = np.where(altitudes_JB2008==alt)
        
        cs= axs[1].contourf(localSolarTimes_JB2008,latitudes_JB2008,JB2008_dens_reshaped[:,:,hi,time_index].squeeze().T)
        axs[1].set_title('JB2008 density at 400 KM, t={} hrs'.format(time_index),fontsize=18)
        axs[1].set_ylabel('Latitudes', fontsize=18)
        axs[1].tick_params(axis='both',which='major',labelsize=16)
        
        cbar= fig.colorbar(cs,ax=axs[1])
        cbar.ax.set_ylabel('Density')
        
        axs[1].set_xlabel('local solar time',fontsize=18)
