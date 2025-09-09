#!/usr/bin/env python3
'''
This code modelise teh atmosphere as 1 layer that:
- absorbs all longwave radiation
- emissivity=absortivity
- transparent to solar radiation
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

def one_layer_T_earth(s0,alpha=0.33,sigma=5.67e-8):
    '''
    This function calculate the Temperature earth would have with a 1 layer (perfectly absorbing and balanced) 
    atmosphere with ratio to the solar irradiance and the Earth Albedo

    Parameters
    ----------

    s0: float:
        The solar constant/ the power per square meter from the sun at the top of the atmosphere

    alpha: floating point, default to 0.33
        The albedo of the earth
    
    sigma: floating point, default to 5.67e-8
        the Boltzmann constant

    Returns
    ---------

    T_e: floating point
        The temperature of Earth's surface in Kelvin
    '''

    T_e=((1-alpha)*s0/(2*sigma))**0.25
    return T_e

def verify_code():
    '''
    Verify that our implementation is correct
    using example problem Dan's video lecture:
    s0=1350 W/m2
    sigma=5.67e-8
    alpha=0.33
     '''
    T_real=298.8
    T_code=one_layer_T_earth(1350)
    print(f"Target solution is {T_real} K")
    print(f"The calculated solution is {T_code} K")
    print(f"The difference is {T_real-T_code}")


#Determine weather variation in Solar irradiance can be a significant cause of Temperature anomaly

year=np.array([1900,1950,2000])
s0=np.array([1365,1366.5,1368])
t_anom=np.array([-0.4,0,0.4])

T_1950=one_layer_T_earth(s0[1])

T_measured=t_anom +T_1950


#Calculation of the modeled values of Earth's surface with the different s0
T_e_calculated=one_layer_T_earth(s0)

#Create our figure and plot 
fig, ax = plt.subplots(1,1)
ax.plot(year, T_measured, label=f"Temperature measured")
ax.plot(year, T_e_calculated, label=f"Temperature modeled only considering s0 changes")

ax.legend()
ax.set_xlabel('Year', fontsize=13)
ax.set_ylabel('Temperature (K)',fontsize=13)
ax.set_title(f"With a one layer atmosphere solar irradiance variation have an insufficient impact \n to explain Earth's surface Temperature on its own",fontsize=15,color='maroon')






