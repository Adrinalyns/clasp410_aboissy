#! /usr/bin/env python3
'''
Lab 05 : Snowball Earth

'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')

#some constants
radearth = 635700 # Earth radius in meters
mxdlyr = 50.      # detpth of mixed layer in meters
sigma = 5.67e-8  # Stefan-Boltzmann constant
C = 4.2e6        # Specific heat capacity of water
rho = 1020       # Density of sea-water (kg/m^3)
alb_ice = 0.6    # Albedo of ice
alb_water = 0.3  # Albedo of water
lam = 100        # Diffusivity of the ocean (m^2/s)

def gen_gird(npoints=18):
    '''
    Create a evenly spaced latitudinal grid with 'npoints' cell centers.
    Grid will always run from 0 to 180 degree as the edges of the grid. This
    means that the first gird point will be 'dLat/2' and the last point will be
    '180 - dLat/2'.

    Parameters
    ----------
    npoints : int, default to 18
        Number of grid points to create.
        
    Returns
    ----------
    dLat : float
        Grid spacing in latitude (degrees).
    lats : Numpy array
        Locations of all grid cell centers (degrees).
    '''
    dlat = 180 / npoints  # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)  # Lat cell centers.

    return dlat, lats

def test_functions():
    ''' 
    Test functions
    '''

    print('Test gen_gird')
    print('For npoints=5:')
    dlat_correct, lats_correct = 36.0, np.array([18., 54., 90., 126., 162.])
    result = gen_gird(5)

    if (result[0] == dlat_correct) and np.all(result[1] == lats_correct):
        print('\tPassed')
    else:
        print('\tFailed')
        print('\tExpected:', (dlat_correct, lats_correct))
        print('\tGot     :', result)


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required.
        0 corresponds to the south pole, 180 to the north.

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])

    # Get base grid:
    npoints = T_warm.size
    dlat, lats = gen_gird(npoints)

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp


def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowball_earth(nlat=18, t_final=10000, dt=1, debug=False):
    '''
    Solve the snowball earth problem.

    Parameters
    ----------
    nlat : int, default to 18
        Number of latitude cells.
    t_final : int or float, default to 10,000
        Time length of simulation in years.
    dt : int or float, default to 1.0
        Size of timestep in years.
    debug : bool, default to False
        If True, print debugging information.

    returns
    ----------
    lats : Numpy array
        Latitudes representing cells centers in degrees. 
        0 is south pole, 180 is north pole.
    temp : Numpy array
        Temperature profile at final time step.

    '''

    #Set number of time steps:
    n_steps = int(t_final/dt)

    #set timestep to seconds:
    dt = dt * 365 * 24.0 * 3600

    # Set up grid:
    dlat, lats = gen_gird(nlat)

    #Set initial temperature:
    T = temp_warm(lats)
    '''
    albedo = np.zeros(nlat)
    frozen = T <= 0.
    albedo[frozen] = alb_ice
    albedo[~frozen] = alb_water
    '''
    K=np.zeros((nlat, nlat))


    for i in range(1,nlat-1):
        for j in range(nlat):
            if i==j:
                K[i, j] = -2.0
            elif(i==j+1 or i==j-1):
                K[i, j] = 1.0

    K[0,0] = -2.0
    K[0,1] = 2.0
    K[nlat-1, nlat-1] = -2.0
    K[nlat-1, nlat-2] = 2.0
    
    L= np.linalg.inv(np.eye(nlat) - dt * lambda_val * K)


    if debug:
        print(K)
    
    for step in range(n_steps):
        T = np.dot(L,T)


    