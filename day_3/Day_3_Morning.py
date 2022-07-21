#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welcome to Space Weather Simulation Summer School Day 3

Today, we will be working with various file types, doing some simple data 
manipulation and data visualization

We will be using a lot of things that have been covered over the last two days 
with minor vairation.

Goal: Getting comfortable with reading and writing data, doing simple data 
manipulation, and visualizing data.

Task: Fill in the cells with the correct codes

@author: Peng Mun Siew
"""

#%% 
"""
This is a code cell that we can use to partition our data. (Similar to Matlab cell)
We hace the options to run the codes cell by cell by using the "Run current cell" button on the top.
"""
print ("Hello World")

#%%
"""
Writing and reading numpy file
"""
# Importing the required packages
import numpy as np

# Generate a random array of dimension 10 by 5
data_arr = np.random.randn(10,5)
print(data_arr)

# Save the data_arr variable into a .npy file
np.save('')
data_arr_loadedd = np.load(test_np_savedata_arr)


print(np.equal(data_arr,data_arr_loadedd))
print(data_arr==data_arr_loadeded)
#%%
"""
Writing and reading numpy zip archive/file
"""
# Generate a second random array of dimension 8 by 1
data_arr2 = np.random.randn(8,1)
print(data_arr2)

# Save the data_arr and data_arr2 variables into a .npz file
np.savez('test_savez.npz',data_arr,data_arr2)


#loaded the numpy zip file
npzfile= np.load('test_savez.npz')
#load file is no numpy array, put is Npzfile object. you ae not able 
#to print the values directly 

print(npzfile)

# to inspect the name of the variables within the npzfile 
print('variable name within this file:',sorted(npzfile.files))

#we will then be able to use the variables name as a key access the data

print(npzfile['arr_0'])

#verification that the loaded data matches 
print((data_arr==npzfile['arr_0']).all())
print((data_arr2==npzfile['arr_1']).all())
#%%
"""
Error and exception
"""



#%%
"""
Loading data from Matlab
"""

# Import required packages
import numpy as np
from scipy.io import loadmat

dir_density_Jb2008 = 'JB2008/2002_JB2008_density.mat'

# Load Density Data
try:
    loaded_data = loadmat(dir_density_Jb2008)
    print (loaded_data)
except:
    print("File not found. Please check your directory")

# Uses key to extract our data of interest
JB2008_dens = loaded_data['densityData']

# The shape command now works
print(JB2008_dens.shape)

#%%
"""
Data visualization I

Let's visualize the density field for 400 KM at different time.
"""
# Import required packages
import matplotlib.pyplot as plt

# Before we can visualize our density data, we first need to generate the discretization grid of the density data in 3D space. We will be using np.linspace to create evenly sapce data between the limits.

localSolarTimes_JB2008 = np.linspace(0,24,24)
latitudes_JB2008 = np.linspace(-87.5,87.5,20)
altitudes_JB2008 = np.linspace(100,800,36)
nofAlt_JB2008 = altitudes_JB2008.shape[0]
nofLst_JB2008 = localSolarTimes_JB2008.shape[0]
nofLat_JB2008 = latitudes_JB2008.shape[0]

# We can also impose additional constratints such as forcing the values to be integers.
time_array_JB2008 = np.linspace(0,8760,20, dtype = int)

# For the dataset that we will be working with today, you will need to reshape them to be lst x lat x altitude
JB2008_dens_reshaped = np.reshape(JB2008_dens,(nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008,8760), order='F') # Fortra-like index order


#%%
"""
Assignment 1

Can you plot the mean density for each altitude at February 1st, 2002?
"""

# First identidy the time index that corresponds to  February 1st, 2002. Note the data is generated at an hourly interval from 00:00 January 1st, 2002
time_index = 31*24
dens_data_feb1 = JB2008_dens_reshaped[:,:,:,time_index]


#%%
"""
Data Visualization II

Now, let's us work with density data from TIE-GCM instead, and plot the density field at 310km
"""
# Import required packages
import h5py




#%%
"""
Data Interpolation (1D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy import interpolate

# Let's first create some data for interpolation
x = np.arange(0, 10)
y = np.exp(-x/3.0)


#%%
"""
Data Interpolation (3D)

Now, let's us look at how to do data interpolation with scipy
"""
# Import required packages
from scipy.interpolate import RegularGridInterpolator

# First create a set of sample data that we will be using 3D interpolation on
def function_1(x, y, z):
    return 2 * x**3 + 3 * y**2 - z

x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)

sample_data = function_1(xg, yg, zg)


#%%
"""
Saving mat file

Now, let's us look at how to we can save our data into a mat file
"""
# Import required packages
from scipy.io import savemat

a = np.arange(20)
mdic = {"a": a, "label": "experiment"} # Using dictionary to store multiple variables
savemat("matlab_matrix.mat", mdic)

#%%
"""
Assignment 2 (a)

The two data that we have been working on today have different discretization grid.
Use 3D interpolation to evaluate the TIE-GCM density field at 400 KM on February 1st, 2002, with the discretized grid used for the JB2008 ((nofLst_JB2008,nofLat_JB2008,nofAlt_JB2008).
"""





#%%
"""
Assignment 2 (b)

Now, let's find the difference between both density models and plot out this difference in a contour plot.
"""





#%%
"""
Assignment 2 (c)

In the scientific field, it is sometime more useful to plot the differences in terms of mean absolute percentage difference/error (MAPE). Let's plot the MAPE for this scenario.
"""





