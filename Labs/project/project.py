#! /usr/bin/env python3
'''
Project : Turning snowball earth model, into a 2D lat-lon model.
Only latitudinal diffusion is considered here.
We will add a heterogenous Anthropogenic Heat Flux, 
and analyze its consequence on global and local climate

To reproduce each plot of this study, write the following instruction in a terminal:
•   ipython
•   run project.py
•   plt.ion()
•   question_1() : to validate the solver by reproducing Figure 1
•   question_2() : to answer Q2
•   question_3() : to answer Q3

'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')

#some constants
radearth = 6357000 # Earth radius in meters
mxdlyr = 50.      # detpth of mixed layer in meters
sigma = 5.67e-8  # Stefan-Boltzmann constant
C = 4.2e6        # Specific heat capacity of water
rho = 1020       # Density of sea-water (kg/m^3)
alb_ice = 0.6    # Albedo of ice
alb_water = 0.3  # Albedo of water
lam = 100        # Diffusivity of the ocean (m^2/s)

def gen_grid(nlats=18,nlongs=36):
    '''
    Create two evenly spaced latitudinal and longitudinal grids with respectively 'nlats' and 'nlongs' cell centers.
    The edges will always run from 0 to 180 degree in latitude and from 0 to 360 degree in longitude.
    This means that the first latitude grid point (center) will be 'dLat/2' and the last point will be at '180 - dLat/2'.
    Similarly, the first longitude grid point (center) will be 'dLong/2' and the last point will be at '360 - dLong/2'.

    Parameters
    ----------
    nlats : int, default to 18
        Number of latitude grid points to create.
    nlongs : int, default to 36
        Number of longitude grid points to create.
        
    Returns
    ----------
    dLat : float
        Grid spacing in latitude (degrees).
    lats : Numpy array
        Locations of all grid cell centers (degrees).
    '''
    dlat = 180 / nlats  # Latitude spacing.
    dlong = 360 / nlongs  # Longitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., nlats)  # Lat cell centers.
    longs = np.linspace(dlong/2., 360-dlong/2., nlongs)  # Long cell centers.

    return dlat, dlong, lats, longs

def test_functions():
    ''' 
    Test functions
    '''

    print('Test gen_grid')
    print('For nlat=5, nlong=6:')
    dlat_correct, lats_correct = 36.0, np.array([18., 54., 90., 126., 162.])
    dlong_correct, longs_correct = 60.0, np.array([30., 90., 150., 210., 270., 330.])
    result = gen_grid(5, 6)

    if (result[0] == dlat_correct) and np.all(result[2] == lats_correct):
        print('\tLatitudinal grid: Passed')
    else:
        print('\tLatitudinal grid: Failed')
        print('\tExpected:', (dlat_correct, lats_correct))
        print('\tGot     :', result)
    if (result[1] == dlong_correct) and np.all(result[3] == longs_correct):
        print('\tLongitudinal grid: Passed')
    else:
        print('\tLongitudinal grid: Failed')
        print('\tExpected:', (dlong_correct, longs_correct))
        print('\tGot     :', result)

    print('Test temp_warm:')
    print('For nlat=5, nlong=6 and same temperature for all longitudes:')

    Temp_correct = np.array([[-21.33, -21.33, -21.33, -21.33, -21.33, -21.33],
                          [ 14.37,  14.37,  14.37,  14.37, 14.37,  14.37],
                          [ 26.27,  26.27,  26.27,  26.27, 26.27,  26.27],
                          [ 14.37,  14.37,  14.37,  14.37, 14.37,  14.37],
                          [-21.33, -21.33, -21.33, -21.33,-21.33, -21.33]])
    temp_grid = temp_warm(result[2], result[3])
    if abs(np.max(temp_grid - Temp_correct)) < 0.01:
        print('\tTemperature grid: Passed')
    else:
        print('\tTemperature grid: Failed')
        print('\tExpected:', Temp_correct)
        print('\tGot     :', temp_grid)

    print('Testing the datacenter_position function:')
    coords = np.array([[0,0],[45,89],[-31,183],[68,275],[90,360],[-90,0]])
    dcpowers = np.array([100e6,200e6,30e6,600e6,150e6,80e6])
    dc_map = datacenter_position(coords,dcpowers,nlat=18,nlong=36,debug=True)
    print('Verify visually that all datacenters are correctly placed on the map with the correct power values.')


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
    dlat, dlong, lats, longs = gen_grid(npoints,1)

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2
    result = temp

    return result


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


def snowball_earth(nlat=18, nlong=36, t_final=10000, dt=1,T_init=temp_warm, apply_sphercorr=False, \
            apply_insol=False, S0=1370, lam=35.0, emiss=0.73, albice=.6, albwater=.3,dc_map=None, debug=False):
    '''
    Solve the snowball earth problem.

    Parameters
    ----------
    nlat : int, default to 18
        Number of latitude cells.
    nlong : int, default to 36
        Number of longitude cells.
    t_final : int or float, default to 10,000
        Time length of simulation in years.
    dt : int or float, default to 1.0
        Size of timestep in years.
    T_init : function or array, default to temp_warm
        Set the initial condition of the simulation. If a function is given,
        it must take latitudes as input and return the array of temperature  
        at specified latitudes. Otherwise the given values are used as-is.
    apply_sphercorr : bool, default to False
        Apply spherical correction term
    apply_insol : bool, default to False
        Apply the radiative transfer
    s0 : float, default to 1370
        Solar constant
    lam : float, default to 100
        Set ocean diffusivity (m^2/s).
    emiss : float, default to 1.0
        Set emissivity of the earth.
    albice, albwater : float, defaul to .6 and .3
        The albedo of ice and water
    dc_map : 2D array, default to None
        A 2D grid (nlat x nlong) with the power of the datacenters in W (it will be converted to W/m^2 over the cell area)
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
    dlat, dlong, lats, longs = gen_grid(nlat, nlong)

    #Set up the coordinates of the cell edges:
    lats_edges = np.linspace(0, 180, nlat+1)
    longs_edges = np.linspace(0, 360, nlong+1)

    # Y-spacing for cells in physical units:
    dy = np.pi * radearth / nlat

    #Set initial temperature:
    Temp_2D = np.zeros((nlat,nlong))
    if callable(T_init):
        for long in range(nlong):
            Temp_2D[:,long] += T_init(lats)
    else:
        Temp_2D += T_init

    #Create our K matrix for diffusion:
    K=np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2.0
    K[np.arange(1,nlat),np.arange(nlat-1)] = 1.0
    K[np.arange(nlat-1),np.arange(1,nlat)] = 1.0
    K[0,1], K[-1,-2] = 2.0, 2.0
    K *= 1.0 / dy**2
    
    L_inv= np.linalg.inv(np.eye(nlat) - dt * lam * K)

    #Create our first derivative operator:
    B=np.zeros((nlat,nlat))
    B[np.arange(1,nlat-1),np.arange(nlat-2)] = -1.0
    B[np.arange(1,nlat-1),np.arange(2,nlat)] = 1.0

    # Create area array:
    Axz = np.pi * ((radearth + 50)**2 - (radearth)**2) * np.sin(np.pi/180.*lats)/nlong
    
    # Get derivative of Area:
    dAxz = np.matmul(B, Axz)

    #Create insolation array
    insol = insolation(S0,lats)

    #Create datacenter power array
    dc_power = np.zeros((nlat,nlong))
    if dc_map is not None:
        Axy = radearth**2 * np.radians(dlong) * (np.cos(np.radians(lats_edges[:-1]))-np.cos(np.radians(lats_edges[1:])))  # Simplified area of each lat-long cell in m^2
        # Divide the power of each datacenter by the area of its cell to get W/m^2
        dc_power =( dc_map.T / Axy ).T  #Allow to divide each column of dc_map by dAxy
    
    # Define the albedo, it will be initialized at the first iteration of the loop
    #Then it will be updated at each time step
    albedo = np.zeros((nlat,nlong))

    if debug:
        print(K)
        print(B)
        plot_temp(dc_power*1000, title='Datacenter power distribution', cbar_label='Power Density $(mW/m^2)$', v_min=0, v_max=5)
        plot_temp(Temp_2D, title='Initial Temperature Profile', v_min=-60, v_max=40)
    
    
    for step in range(n_steps):
        for lon in range(nlong):
            T = Temp_2D[:,lon]

            #Update albedo:
            loc_ice = T < -10.
            albedo[loc_ice] = albice
            albedo[~loc_ice] = albwater

            # Create spherical coordinate correction term:
            sphercorr = int(apply_sphercorr) * (lam * dt)/(4 *Axz * dy**2) * np.matmul(B, T) * dAxz

            #Calculate vertical heat fluxes 
            vertical =  dt/(rho*C*mxdlyr) * ( int(apply_insol)*((1-albedo[:,lon])*insol - emiss * sigma * (T+273)**4) + dc_power[:,lon] )
 
            T = np.matmul(L_inv,T + sphercorr + vertical)
            Temp_2D[:,lon] = T
    
        if debug:
            if step % 2000 == 0:
                print('At time step ', step, ' / ', n_steps)
                plot_temp(Temp_2D, title=f'Temperature Profile at step {step}', v_min=-60, v_max=40)
    
    return lats, longs, Temp_2D


def plot_temp(Temp_2D, title='Temperature Profile', cbar_label='Temperature (°C)', v_min=-60, v_max=40, numbers=True, **kwargs):
    """
    Plot the temperature across the globe, by using a 2D colormap.
    the color represents the temperature: a diverging colormap is used, 
    and the diverging point is set at -10°C to make the distinction 
    between frozen and ice free areas.

    Parameters
    ------------
    Temp_2D : 2D array
        The 2D array containing the temperatures to be plotted across the globe
    title : string
        The title of the generated graph
    cbar_label : string
        The label of the colorbar
    v_min, v_max : float, default to -60, 40
        The minimal and maximal value of the colorbar use to plot the colormap
    numbers : bool
        If True, the value of each cell is plotted inside the cell
    
    Returns
    ------------
    fig, ax : figure, ax
        It returns the figure and the ax so that it could be reused

    """
    nlat, nlong = Temp_2D.shape
    lats_edges = np.linspace(0, 180, nlat+1)
    longs_edges = np.linspace(0, 360, nlong+1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    contour =plt.pcolormesh(longs_edges, lats_edges-90., Temp_2D, shading='auto', cmap='seismic', vmin=v_min, vmax=v_max, **kwargs)
    fig.colorbar(contour, ax=ax, label=cbar_label, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title(title)
    if numbers:
        for i in range(len(lats_edges)-1):
            for j in range(len(longs_edges)-1):
                ax.text((longs_edges[j] + longs_edges[j+1])/2, (lats_edges[i] + lats_edges[i+1])/2 - 90., f'{Temp_2D[i,j]:.1f}', ha='center', va='center', color='black', fontsize=6)
    plt.show()

    return fig,ax


def anthropogenic_power_map(coords,dcpowers,nlat=18,nlong=36,debug=False):
    """
    Given a list of coordinates (latitude, longitude) in degrees representing the position of datacenters
    and a list of their respective power in W,
    This function creates a 2D map of the earth (nlat x nlong), 
    and places the datacenters total power at the closest grid points to the given coordinates.

    Parameters
    ------------
    nlat : int, default to 18
        Number of latitude cells.
    nlong : int, default to 36
        Number of longitude cells.
    coords : np.array 
        The coordinates of each datacenter
    dcpowers :  np.array
        The power of each datacenter
    debug : bool
        If True, the function prints the power map that it has generated 
        and the original position of the heat sources

    Returns
    ------------
    dc_map : 2D array
        A 2D grid (nlat x nlong) with the power of each datacenter in W 

    """
    dc_map=np.zeros((nlat,nlong))

    dlat = 180/nlat
    dlong = 360/nlong

    lat_pos = ((coords[:,0]+90)//(dlat+0.0001)).astype(int) #Adding a very small number to dlat, so that when the latitude is exactly 90,
                                                            # it does not go out of bounds
    long_pos = (coords[:,1]//(dlong+0.0001)).astype(int)    #Adding a very small number to dlong, so that when the latitude is exactly 360,
                                                            # it does not go out of bounds

    for index_dc in range(coords.shape[0]):
        dc_map[lat_pos[index_dc],long_pos[index_dc]]+=dcpowers[index_dc]

    #Debug part: it prints the AHF map creating and the coordinate of major city to see if they match the cells
    if debug:
        fig,ax = plot_temp(dc_map/1e6, title='Datacenter power distribution', cbar_label='Anthropogenic Power $(MW)$', v_min=-700., v_max=700.)
        
        for index,(x,y) in enumerate(coords):
            ax.plot(y,x,'+',color='lime',markersize=15)
            ax.text(y+3,x+5,f"{dcpowers[index]/1e6:.1f} MW",color='lime',weight='bold')

    return dc_map


def question_1():
    """
    Reproducing Figure 1:
    Equilibrium temperature over the globe without AHF
    """
    nlats = 18
    nlongs = 36
    lats, longs, Temp_final = snowball_earth(nlat=nlats, nlong=nlongs, t_final=10000, dt=1, T_init=temp_warm, \
                                            apply_sphercorr=True, apply_insol=True, S0=1370, lam=35.0, emiss=0.73, \
                                            albice=.6, albwater=.3)
    plot_temp(Temp_final,title='Final Temperature Profile without Anthopogenic Heat Flux', v_min=-60, v_max=40, numbers=True)


def question_2():
    '''
    Verifying the anthropogenic_power_map() function, and analyzing the warming caused by the 12 biggest datacenters
    '''

    nlats = 18
    nlongs = 36

    #12 biggest datacenters
    coords = np.array([[50,9],[40,353],[40,269],[19,73],[42,272],[34,276],\
                        [52,357],[41,264],[36,245],[40,117],[46,127],[39,240],[41,112]])
    dcpowers = np.array([60, 12, 40, 50, 100, 100, 148, 100, 315, 150, 200, 650,150])*1e6  # Powers in W
    
    #Validating anthropogenic_power_map()
    dc_map = anthropogenic_power_map(coords,dcpowers,nlat=18,nlong=36,debug=True)

    #Analyzing the temperature equilibrium with the AHF
    lats, longs, Temp_final = snowball_earth(nlat=nlats, nlong=nlongs, t_final=10000, dt=1, T_init=temp_warm, \
                                            apply_sphercorr=True, apply_insol=True, S0=1370, lam=35.0, emiss=0.73, \
                                            albice=.6, albwater=.3,dc_map=dc_map)
    plot_temp(Temp_final, title=f'Final Temperature Profile with the 12 biggest datacenters', v_min=-60, v_max=40, numbers=True)
    

def question_3():
    '''
    Considering a total power of 35 GW distributed over the world, 
    see how the configuration of datacenters affects the final temperature profile.
    '''
    P_tot = 16.6e12  # Total power in W
    nlats = 30
    nlong_array = np.array([18,72,126,180])
    lat_city = 40
    kwargs = {'nlat':nlats, 't_final':10000, 'dt':1, 'T_init':temp_warm, \
              'apply_sphercorr':True, 'apply_insol':True, 'S0':1370, 'lam':35.0, 'emiss':0.73, \
              'albice':.6, 'albwater':.3, 'debug':False}


    coords = np.array([ [lat_city,18],[lat_city,54],[lat_city,90],[lat_city,126],[lat_city,162],\
                    [lat_city,198],[lat_city,234],[lat_city,270],[lat_city,306],[lat_city,342] ])
    powers = np.zeros(10) + P_tot/10.

    mean_temp_array = np.zeros(len(nlong_array))

    for index, nlongs in enumerate(nlong_array): 
        kwargs['nlong']=nlongs
        dc_map = anthropogenic_power_map(coords,powers,nlat= nlats, nlong=nlongs,debug=False)
        
        kwargs['dc_map']=dc_map
        lats, longs, Temp_final = snowball_earth(**kwargs)
        plot_temp(Temp_final, title=f'Final Temperature Profile with 10 Megalopolis and {nlongs} longitudes', v_min=-60, v_max=40, numbers=False)