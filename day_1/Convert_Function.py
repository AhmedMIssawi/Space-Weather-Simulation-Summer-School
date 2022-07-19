# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:16:43 2022

@author: Ahmed Issawi

Ahmed Issawi 
aissawi@unb.ca

A 3D plot 
"""
import numpy as np 

def convert_function(r,phi,zta):
    """
    converting from the spherical co-ordinates to cartisian co-ordinates
    using numerical library numpys to compute the sin and cosin.
    
    parameters
    ----------
    r : integer
        radial.
    phi : angle
        zentith.
    zta : angle
        azimuthal.

    Returns
    -------
    dict_result : inetger
        X, Y, Z the location position.
        
    Example: 
        convert_function(2,np.pi,60)
        the r = 2 
        the phi = Pi
        zta = 60 
        
    """
    
    x = r * np.sin(phi) * np.cos(zta)
    y = r * np.sin(phi) * np.sin(zta)
    z = r * np.cos(phi)
    
    dict_result = {'x': x,
                   'y': y,
                   'z': z
                   }
    
    return dict_result

print(convert_function(2,np.pi,60))

import matplotlib.pyplot as plt 
if __name__ == '__main__':
    fig = plt.figure()
    axes = fig.gca(projection='3d')
    r = np.linspace(0,1)
    theta = np.linspace(0,2*np.pi)
    phi = np.linspace(0,2*np.pi)
    coords = convert_function(r, theta, phi)
    axes.plot(coords['x'],coords['y'],coords['z'])
