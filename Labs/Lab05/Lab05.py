#! /usr/bin/env python3
'''
Lab 05 : Snowball Earth

author : Adrien Boissy
date   : Fall 2025
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


def snowball_earth(nlat=18, t_final=10000, dt=1,T_init=temp_warm, apply_sphercorr=False, apply_insol=False, S0=1370, lam=100.0, emiss=1.0, albice=.6, albwater=.3, debug=False):
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

    # Y-spacing for cells in physical units:
    dy = np.pi * radearth / nlat

    #Set initial temperature:
    T = np.zeros(nlat)
    if callable(T_init):
        T = T_init(lats)
    else:
        T += T_init

    '''
    albedo = np.zeros(nlat)
    frozen = T <= 0.
    albedo[frozen] = alb_ice
    albedo[~frozen] = alb_water
    '''

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
    Axz = np.pi * ((radearth + 50)**2 - (radearth)**2) * np.sin(np.pi/180.*lats)

    # Get derivative of Area:
    dAxz = np.matmul(B, Axz)

    #Create insolation array
    insol = insolation(S0,lats)
    
    # Define the albedo, it will be initialized at the first iteration of the loop
    #Then it will be updated at each time step
    albedo = np.zeros(nlat)

    if debug:
        print(K)
        print(B)
    
    for step in range(n_steps):
        #Update albedo:
        loc_ice = T < -10.
        albedo[loc_ice] = albice
        albedo[~loc_ice] = albwater

        # Create spherical coordinate correction term:
        sphercorr = int(apply_sphercorr) * (lam * dt)/(4 *Axz * dy**2) * np.matmul(B, T) * dAxz

        #Calculate insolation term
        radiative = int(apply_insol) * dt/(rho*C*mxdlyr) * ((1-albedo)*insol - emiss * sigma * (T+273)**4)

        T = np.matmul(L_inv,T + sphercorr + radiative)

    return lats, T


def question_1():
    '''' 
    Create solution figure for testing the snowball earth model and answering question 1.
    '''
    # Get solution after 10K years, diffusion only
    lats, T_final = snowball_earth()

    # Get solution after 10K years, diffusion and spherical correction
    lats, T_sphercorr = snowball_earth(apply_sphercorr=True)

    # Get solution after 10K years, diffusion, spherical correction and radiative transfer
    lats, T_all = snowball_earth(apply_sphercorr=True, apply_insol=True, albice=.3)

    # Get the initial condition
    T_initial = temp_warm(lats)

    #Create a fancy plot:

    fig,ax = plt.subplots(1,1)
    ax.plot(lats - 90., T_initial, label='Initial Condition', color='deepskyblue', linewidth=4)
    ax.plot(lats - 90., T_final, label='Diffusion',color='red', linewidth=4)
    ax.plot(lats - 90., T_sphercorr, label='Diffusion + Spherical Corr.', color='orange', linewidth=4)
    ax.plot(lats - 90., T_all, label='Diffusion + Spherical Corr. + Radiative', color='olivedrab', linewidth=4)

    # Customize the plot:
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Solution after 10,000 Years')
    ax.legend(loc='best')

def question_2():
    '''
    Answer question 2 by testing a range of thermal diffusivities, and emissivities,
    to find a combination that allow the warm state to persist.
    '''
    kwargs={'nlat':40, 't_final':10000, 'dt':1, 'apply_sphercorr':True, 'apply_insol':True, \
            'S0':1370, 'lam': 75, 'emiss': 0.5, 'albice':.3, 'albwater':.3, 'debug':False}
    
    fig, ax = plt.subplots(1,2, figsize=(13,8))

    #Varying the thermal diffusivity of the ocean
    lam_values = np.linspace(0,150,6)
    for lam in lam_values:
        kwargs['lam']=lam
        lats, T_final = snowball_earth(**kwargs)
        ax[0].plot(lats - 90., T_final, label=f'$\lambda$ = {lam:.1f} $W/m^2 K$')
    
    T_warm = temp_warm(lats)
    ax[0].plot(lats - 90., T_warm, label='Warm Earth', color='red', linewidth=3)

    ax[0].set_xlabel('Latitude (°)')
    ax[0].set_ylabel('Temperature (°C)')
    ax[0].set_title(f'Varying Thermal Diffusivity of Ocean\n $\epsilon$ = {kwargs["emiss"]:.2f}')
    ax[0].legend(loc='best')

    #Restore lambda value
    kwargs['lam']=75.0

    #Varying the emissivity of Earth
    emiss_values = np.linspace(0.2,1,6)
    for emiss in emiss_values:
        kwargs['emiss']=emiss
        lats, T_final = snowball_earth(**kwargs)
        ax[1].plot(lats - 90., T_final, label=f'$\epsilon$ = {emiss:.2f}')

    #Plotting the warm earth solution for reference
    ax[1].plot(lats - 90., T_warm, label='Warm Earth', color='red', linewidth=3)
  
    ax[1].set_xlabel('Latitude (°)')
    ax[1].set_ylabel('Temperature (°C)')
    ax[1].set_title(f'Varying Emissivity of Earth\n $\lambda$ = {kwargs["lam"]:.2f} $W/m^2 K$')
    ax[1].legend(loc='best')
    
    lam_opt = 35.0
    emiss_opt = 0.73

    #Calculating the optimal solution: keeping earth temperature as it is today
    kwargs['lam']=lam_opt
    kwargs['emiss']=emiss_opt
    lats, T_final = snowball_earth(**kwargs)

    #Plotting the optimal solution
    fig2,ax2 = plt.subplots(1,1)
    ax2.plot(lats - 90., T_warm, label='Initial State: warm Earth', color='red', linewidth=4)
    ax2.plot(lats - 90., T_final, label='Final State', color='black', linewidth=2)
    ax2.set_xlabel('Latitude (°)')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title(f'To keep the earth as it is today, we need \n \
                    $\lambda$ = {lam_opt:.2f} $W/m^2 K$ and $\epsilon$ = {emiss_opt:.2f}')
    ax2.legend(loc='best')
    
    plt.show()

def question_3():
    '''
    Answer question 3 by testing different initial conditions, 
    and looking at their effect on the equilibrium state.
    '''

    #Calculating the equilibrium with a hot earth initial condition
    T_initial_hot = 60. #°C
    kwargs={'nlat':90, 't_final':10000, 'dt':1, 'T_init': T_initial_hot, 'apply_sphercorr':True, \
            'apply_insol':True, 'S0':1370, 'lam': 35.0, 'emiss': 0.73, 'albice':.6, 'albwater':.3, 'debug':False}
    lats, T_final_hot = snowball_earth(**kwargs)
    loc_ice_free = T_final_hot > -10.

    #Calculating the equilibrium with a warm earth initial condition -> warm equilibrium
    T_initial_warm = temp_warm(lats) #°C
    kwargs['T_init'] = T_initial_warm
    lats, T_final_warm = snowball_earth(**kwargs)
    loc_ice_free = T_final_warm > -10.

    fig,ax = plt.subplots(1,2, figsize=(14,7))

    #Plotting the hot earth initial condition and the resulting equilibrium
    ax[0].plot([-90.,90.],[T_initial_hot,T_initial_hot], label='Initial Hot Temperature', color='red')
    ax[0].plot(lats - 90., T_final_hot, label=f'Hot Earth equilibrium: Frozen regions',color='cornflowerblue')
    ax[0].plot(lats[loc_ice_free] - 90., T_final_hot[loc_ice_free], label='Hot Earth equilibrium: Ice free regions', color='maroon')
    
    #Plotting the warm earth equilibrium for comparison
    ax[0].plot(lats - 90., T_final_warm, label='Warm Earth Equilibrium: Frozen regions', color='cornflowerblue',linestyle='dotted')
    ax[0].plot(lats[loc_ice_free] - 90., T_final_warm[loc_ice_free], label='Warm Earth equilibrium: Ice free regions', color='maroon', linestyle='dotted')
    
    #Customize the plot
    ax[0].set_xlabel('Latitude (°)')
    ax[0].set_ylabel('Temperature (°C)')
    ax[0].set_title('When the there is no ice, the earth comes back to the warm state')
    ax[0].legend(loc='best')
    
    #Calculating the equilibrium with a cold earth initial condition
    T_initial_cold = -60. #°C
    kwargs['T_init'] = T_initial_cold
    lats, T_final_cold = snowball_earth(**kwargs)

    #Plotting the cold earth solution
    ax[1].plot([-90.,90.],[T_initial_cold,T_initial_cold], label='Initial Cold Temperature', color='mediumblue')
    ax[1].plot(lats - 90., T_final_cold, label=f'Snowball Earth equilibrium', color='cornflowerblue')
    #Customize the plot
    ax[1].set_xlabel('Latitude (°)')
    ax[1].set_ylabel('Temperature (°C)')
    ax[1].set_title('When the whole earth is frozen, it cannot escape the snowball state')
    ax[1].legend(loc='best')

    #Warm earth, with flash freeze initial condition: the albedo is set to ice albedo everywhere
    T_initial_warm = temp_warm(lats)
    kwargs['T_init'] = T_initial_warm
    #artificially set albedo to ice albedo everywhere
    kwargs['albwater'] = .6
    lats, T_final_flash_freeze = snowball_earth(**kwargs)

    #Plotting the flash freeze equilibrium
    fig2,ax2 = plt.subplots(1,1)
    ax2.plot(lats - 90., T_initial_warm, label='Initial Warm Temperature', color='orange')
    ax2.plot(lats - 90., T_final_flash_freeze, label='Flash Freeze equilibrium', color='cornflowerblue')

    #Plotting the snowball earth equilibrium for comparison
    ax2.plot(lats - 90., T_final_cold, label='Snowball Earth equilibrium', color='cornflowerblue', linestyle='dotted')

    #Plotting the warm earth equilibrium for comparison
    ax2.plot(lats - 90., T_final_hot, label='Warm Earth Equilibrium', color='red', linestyle='dotted')

    #Customize the plot
    ax2.set_xlabel('Latitude (°)')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('If the whole earth is suddenly frozen (still hot),\n it reaches the snowball state')
    ax2.legend(loc='best')
    
    plt.show()

def question_4():
    '''
    Answer question 4 by testing the influence of slow increase/decrease of the solar constant.
    '''

    gamma_init= 0.4 #Initial fraction of solar constant

    gamma_values_rise = np.arange(gamma_init, 1.45, 0.05)

    #Initial cold earth condition
    T_init = -60. #°C

    #Parameters for the simulation
    kwargs={'nlat':90, 't_final':10000, 'dt':1, 'T_init':T_init, 'apply_sphercorr':True, 'apply_insol':True, \
            'S0':gamma_init*1370, 'lam': 35, 'emiss': 0.73, 'albice':.6, 'albwater':.3, 'debug':False}

    #Defining a vector of the average temperature at equilibrium for each gamma value
    T_avg_eq_rise = []

    for gamma in gamma_values_rise:
        kwargs['S0'] = gamma * 1370
        kwargs['T_init'] = T_init
        lats, T_final = snowball_earth(**kwargs)
        T_avg_eq_rise.append(np.mean(T_final))
        T_init = T_final #Use the final state as initial condition for the next gamma value
    fig,ax = plt.subplots(1,1)
    ax.plot(gamma_values_rise, T_avg_eq_rise, marker='o')

    gamma_values_drop = np.arange(1.4, gamma_init-0.05, -0.05)
    #Defining a vector of the average temperature at equilibrium for each gamma value
    T_avg_eq_drop = []

    for gamma in gamma_values_drop:
        kwargs['S0'] = gamma * 1370
        kwargs['T_init'] = T_init
        lats, T_final = snowball_earth(**kwargs)
        T_avg_eq_drop.append(np.mean(T_final))
        T_init = T_final #Use the final state as initial condition for the next gamma value
    fig,ax = plt.subplots(1,1)
    ax.plot(gamma_values_drop, T_avg_eq_drop, marker='o')
        




