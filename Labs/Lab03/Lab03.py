#!/usr/bin/env python3
'''
Author: Adrien Boissy
Collaborators: None

'''

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-darkgrid')

S_IN_YEAR=365.25*24*3600
S_IN_DAY=24*3600
DAY_IN_YEAR=365.25

def bar_initial_T(x):
    '''
    Initial temperature distribution in the bar

    Parameters:
    -----------
    x : Numpy array
        A position in the bar

    Returns:
    ---------
    T : Numpy array
        Initial temperature of the bar at the position x
    '''
    return 4*x - 4*x**2

def initial_0(x):
    '''
    Function that return the Initial temperature in the Kangerlussuaq ground as 0 °C everywhere

    Parameters:
    -----------
    x : Numpy array
        The soil depth (m)

    Returns:
    ---------
    T : Numpy array
        Initial temperature of the soil at the depth x (°C)
    '''
    return 0

def temp_kanger(t):
    '''
    This function returns the temperature (°C) at the surface in Kangerlussuaq as a function of time (s).

    Parameters:
    -----------
    t : float
        Time in seconds

    Returns:
    ---------
    temp : float
        Temperature in degree Celsius at the Surface in Kangerlussuaq
    '''

    # Kangerlussuaq average temperature:
    t_kanger = np.array([-19.7,-21.0,10.7, 8.5, 3.1,-17.,-6.0,-8.4, 2.3, 8.4,-12.0,-16.9])
    
    t_amp = (t_kanger - t_kanger.mean()).max()
    omega=360*t/S_IN_YEAR # conversion of the time in second into a fraction of period, that represent the fraction of the year
    return t_amp*np.sin(np.pi/180 * omega - np.pi/2) + t_kanger.mean()

def test_temp_surface(temperature_surface=temp_kanger):
    '''
    Test the temp_kanger function by plotting the temperature over 2 years.
    '''
    t_max=2*S_IN_YEAR
    t=np.arange(0,t_max,3600)

    Temperatures=temperature_surface(t)
    fig,ax=plt.subplots(1,1)
    ax.plot(t/S_IN_YEAR,Temperatures)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Kangerlussuaq Surface Temperature over 2 years')
    plt.show()

def temp_kanger_0_5(t):
    '''
    This function returns the temperature (°C) at the surface in Kangerlussuaq as a function of time (s)
    with 0.5°C of mean global warming.

    Parameters:
    -----------
    t : float
        Time in seconds

    Returns:
    ---------
    temp : float
        Temperature in degree Celsius at the Surface in Kangerlussuaq
    '''

    # Kangerlussuaq average temperature:
    t_kanger = np.array([-19.7,-21.0,10.7, 8.5, 3.1,-17.,-6.0,-8.4, 2.3, 8.4,-12.0,-16.9])
    
    t_amp = (t_kanger - t_kanger.mean()).max()
    omega=360*t/S_IN_YEAR # conversion of the time in second into a fraction of period, that represent the fraction of the year
    return t_amp*np.sin(np.pi/180 * omega - np.pi/2) + t_kanger.mean() +0.5

def temp_kanger_1(t):
    '''
    This function returns the temperature (°C) at the surface in Kangerlussuaq as a function of time (s)
    with 1°C of mean global warming.

    Parameters:
    -----------
    t : float
        Time in seconds

    Returns:
    ---------
    temp : float
        Temperature in degree Celsius at the Surface in Kangerlussuaq
    '''

    # Kangerlussuaq average temperature:
    t_kanger = np.array([-19.7,-21.0,10.7, 8.5, 3.1,-17.,-6.0,-8.4, 2.3, 8.4,-12.0,-16.9])
    
    t_amp = (t_kanger - t_kanger.mean()).max()
    omega=360*t/S_IN_YEAR # conversion of the time in second into a fraction of period, that represent the fraction of the year
    return t_amp*np.sin(np.pi/180 * omega - np.pi/2) + t_kanger.mean() +1

def temp_kanger_3(t):
    '''
    This function returns the temperature (°C) at the surface in Kangerlussuaq as a function of time (s)
    with 3°C of mean global warming.

    Parameters:
    -----------
    t : float
        Time in seconds

    Returns:
    ---------
    temp : float
        Temperature in degree Celsius at the Surface in Kangerlussuaq
    '''

    # Kangerlussuaq average temperature:
    t_kanger = np.array([-19.7,-21.0,10.7, 8.5, 3.1,-17.,-6.0,-8.4, 2.3, 8.4,-12.0,-16.9])
    
    t_amp = (t_kanger - t_kanger.mean()).max()
    omega=360*t/S_IN_YEAR # conversion of the time in second into a fraction of period, that represent the fraction of the year
    return t_amp*np.sin(np.pi/180 * omega - np.pi/2) + t_kanger.mean() +3



def solve_heat(c2=1,x_init=0,x_final=1,dx=0.02,x_array_init=bar_initial_T,t_init=0,t_final=0.2,dt=0.0002,lowerbound=0,upperbound=0):
    """
    Solve the Heat equation for a bar with Neumann boundary conditions, and an initial condition of Ti(x)=4x-4x^2

    Parameters:
    ----------
    c2: float
        c^2, the square of the diffusion coefficient
    x_init, x_final : float
        The initial and final space values, they are included in the grid
    dx : float
        The space step
    x_array_init : function or array
        The initial temperature along the bar, it should be a function of x or an array of size M
    t_init, t_final : float
        The initial and final time values, they are included in the grid
    dt : float
        The time step
    T_border : float, default 0
        The temperature at the borders of the grid
    lowerbound : float or function or None, default 0
        The Dirichlet boundary condition at x_init if not None (otherwise Neumann BC with a zero gradient)
    upperbound : float or function or None, default 0
        The Dirichlet boundary condition at x_final if not None (otherwise Neumann BC with a zero gradient)

    returns:
    ---------
    x,t : 1D Numpy arrays
        Space and time values, respectively
    U : Numpy array
        The solution of the heat equation, size is N (space) x M (time)

    """

    #check stability criteria
    dt_max = dx**2 / (2*c2)
    if (dt > dx**2 /(2*c2)):
        raise ValueError(f'DANGER: dt={dt} > dt_max={dt_max}.')

    N=int((t_final-t_init)/dt)+1
    M=int((x_final-x_init)/dx)+1
    #Set up space and time grid
    t = np.linspace(t_init,t_final,N)
    x = np.linspace(x_init,x_final,M)
    #Create solution matrix and set initial condition
    U=np.zeros([M,N])
    if callable(x_array_init):
        U[:,0]=x_array_init(x)
    else:
        U[:,0]=x_array_init

    #Get the r coeff
    r=c2*(dt/dx**2)
    #Solve equation
    for j in range(N-1):
        U[1:M-1,j+1] = (1-2*r)*U[1:M-1,j]+ r*(U[2:M,j]+ U[:M-2,j])
        if lowerbound is None:
            U[0,j+1] = U[1,j+1] #Neumann boundary condition at x=x_init
        elif callable(lowerbound):
            U[0,j+1] = lowerbound(t[j+1]) #Dirichlet boundary condition function of time
        else:
            U[0,j+1] = lowerbound #Dirichlet constant boundary condition
    
        if upperbound is None:
            U[-1,j+1] = U[-2,j+1] #Neumann boundary condition at x=x_final
        elif callable(upperbound):
            U[-1,j+1] = upperbound(t[j+1]) #Dirichlet boundary condition function of time
        else:
            U[-1,j+1] = upperbound #Dirichlet constant boundary condition
    return x,t,U

def test_solve_heat():
    '''
    This function tests the solve_heat function against a known solution.
    The parameters are :
    c2=1 m^2/s, 
    x_init=0 m, x_final=1 m, dx=0.2 m
    t_init=0 s, t_final=0.2 s, dt=0.02 s
    U(x,0)= 4x-4x^2
    lowerbound = upperbound = 0 °C

    To verify that the two solution are close enough, the function checks that 
    the maximum error between the two solution is below 1e-3, if not an assertion error is raised.
    If the test is passed, a message is printed with the maximum error found.
    '''
    c2=1
    x_init=0
    x_final=1
    dx=0.2
    t_init=0
    t_final=0.2
    dt=0.02
    lowerbound = 0
    upperbound = 0

    #Solving the heat equation in the given situation with our solover function : solve_heat()
    t,x,U=solve_heat(c2=c2,x_init=x_init,x_final=x_final,dx=dx,t_init=t_init,t_final=t_final,dt=dt,lowerbound=lowerbound,upperbound=upperbound)

    # Solution to problem 10.3 from fink/matthews as a nested list:
    sol10p3 = [[0.000000, 0.640000, 0.960000, 0.960000, 0.640000, 0.000000],
           [0.000000, 0.480000, 0.800000, 0.800000, 0.480000, 0.000000],
           [0.000000, 0.400000, 0.640000, 0.640000, 0.400000, 0.000000],
           [0.000000, 0.320000, 0.520000, 0.520000, 0.320000, 0.000000],
           [0.000000, 0.260000, 0.420000, 0.420000, 0.260000, 0.000000],
           [0.000000, 0.210000, 0.340000, 0.340000, 0.210000, 0.000000],
           [0.000000, 0.170000, 0.275000, 0.275000, 0.170000, 0.000000],
           [0.000000, 0.137500, 0.222500, 0.222500, 0.137500, 0.000000],
           [0.000000, 0.111250, 0.180000, 0.180000, 0.111250, 0.000000],
           [0.000000, 0.090000, 0.145625, 0.145625, 0.090000, 0.000000],
           [0.000000, 0.072812, 0.117813, 0.117813, 0.072812, 0.000000]]

    # Convert to an array and transpose it to get correct ordering:
    sol10p3 = np.array(sol10p3).transpose()

    # Check that our solution is close to the known solution:
    #Verify that the two matrices have the same shapes
    assert (U.shape == sol10p3.shape), f'the shape of the solution is incorrect, it is {U.shape} while it should be {sol10p3.shape}'
    N,M=U.shape

    max_error_allowed=1e-3 #Maximum error allowed to pass the test
    max_error=0 #Maximum error found in the test
    
    #Verify that each value of the matrices are clos enough
    for i in range(N):
        for j in range(M):
            assert(abs(U[i,j]-sol10p3[i,j])<max_error_allowed), f'at the position {x[i]} and the time {t[j]}, ' + \
            f'\nthe difference between the computed solution and the known solution is too high : \n error of {abs(U[i,j]-sol10p3[i,j]):.4f}' + \
            f' higher than {max_error_allowed}'

            #Change the maximum error found if the current error is higher
            if abs(U[i,j]-sol10p3[i,j])>max_error:
                max_error=abs(U[i,j]-sol10p3[i,j])

    print(f'Test passed ! \n the calculated solution is close to the known solution with a maximum error of {max_error:.6f} (max allowed is {max_error_allowed})')

def question_1():
    '''
    This function answers question 1.
    It test if the solve_heat function gives the ewpected result in a certain case,
    when we know the expected result.
    Then, it plot the result.
    '''
    
    test_solve_heat()

    c2=1
    x_init=0
    x_final=1
    dx=0.2
    t_init=0
    t_final=0.2
    dt=0.02
    lowerbound = 0
    upperbound = 0

    x,t,U=solve_heat(c2=c2,x_init=x_init,x_final=x_final,dx=dx,t_init=t_init,t_final=t_final,dt=dt,lowerbound=lowerbound,upperbound=upperbound)
    plot_heatsolve(x,t,U,'Temperature diffusion in a bar with Dirichlet boundary conditions')

def question_2():
    '''
    This function answers question 2.
    '''

    Kangerlussuaq_T_variation(5)
    Kangerlussuaq_T_variation(70)
    
def question_3():
    '''
    This function answers question 3.
    '''
    #Scenario 3: 3°C of global warming
    #Calculating the temperature profile after 100 years, to avoid waiting for convergence when plotting the temperature
    x,t,U=Kangerlussuaq_T_variation(100,temp_surface=temp_kanger_3,initial_T=initial_0,plotting=False)
    T_convergence_3 = U[:,-1]
    Kangerlussuaq_T_variation(10,temp_surface=temp_kanger_3,complementary_title=' with 3°C of climate warming',initial_T=T_convergence_3,plotting=True)
 
    #Scenario 2: 1°C of global warming
    #Calculating the temperature profile after 100 years, to avoid waiting for convergence when plotting the temperature
    x,t,U=Kangerlussuaq_T_variation(100,temp_surface=temp_kanger_1,initial_T=initial_0,plotting=False)
    T_convergence_1 = U[:,-1]
    Kangerlussuaq_T_variation(10,temp_surface=temp_kanger_1,complementary_title=' with 1°C of climate warming',initial_T=T_convergence_1,plotting=True)
    
    #Scenario 1: 0.5°C of global warming
    #Calculating the temperature profile after 100 years, to avoid waiting for convergence when plotting the temperature
    x,t,U=Kangerlussuaq_T_variation(100,temp_surface=temp_kanger_0_5,initial_T=initial_0,plotting=False)
    T_convergence_0_5 = U[:,-1]
    Kangerlussuaq_T_variation(10,temp_surface=temp_kanger_0_5,complementary_title=' with 0.5°C of climate warming',initial_T=T_convergence_0_5,plotting=True)


def Kangerlussuaq_T_variation(nb_years,temp_surface=temp_kanger,complementary_title='',initial_T=initial_0,plotting=True):
    '''
    This function calculates and plot the temperature variation in the Kangerlussuaq soil over a given number of years, 
    for a given surface temperature variation and a given initial temperature profile.
    It plots the temperature over time and space in a 2D colormap,
    and the temperature profile in the soil during summer and winter (and permafrost layers)
    after the given number of years.

    Parameters:
    -----------
    nb_years : int
        The number of years to simulate
    temp_surface : function, default temp_kanger
        The function that gives the surface temperature variation over time
    complementary_title : string, default ''
        A complementary title to add on the plots (indicating for example the climate warming scenario, or the initial temperature profile)
    initial_T : function or array, default initial_0
        The initial temperature profile in the soil, it should be a function of depth or an array of size M
    plotting : bool, default True
        If True, the function will plot the results. If False, it will only return the results.
   
    Returns:
    ---------
    x,t : 1D arrays
        Space and time scales
    U : 2D array
        The temperature in the soil for each depth and time (x,t)
    '''
    #defining the parameters of the resolution
    c2=0.25e-6 #m2/s
    x_init=-100 #m
    x_final=0 #m
    dx=0.1 #m
    t_init=0 #s
    t_final=nb_years*S_IN_YEAR #s
    dt=1/5.*S_IN_DAY #s
    lowerbound=5 #°C
    upperbound=temp_surface #callable

    #Solving the Temperature using solve_heat()
    x,t,U=solve_heat(c2=c2,x_init=x_init,x_final=x_final,dx=dx,x_array_init=initial_T,t_init=t_init,t_final=t_final,dt=dt,lowerbound=lowerbound,upperbound=upperbound)
    
    if plotting:
        #Converting time from seconds to years
        t=t/S_IN_YEAR

        #Choosing the best colorbar
            #We want a diverging colormap centered on 0 °C, I chose the 'seismic' colormap
            # Because it was a good visual representation of the temperature above and under 0°C
            #And it was the only one that would allow to well the small variations around 0°C
        colormap='seismic'
            #To ensure that the True zero is at the center of the colormap, we need to center the vmin and vmax around 0
            #We can see in the function temp_kanger that the temperature at the surface oscillate between around -22°C and 11°C
            #So we can choose a vmax of 22°C and a vmin of -22°C to center the colormap around 0°C
        vmax=23
        vmin=-23
        
        #Plotting the temperature variation in the soil over time
        plot_heatsolve(x,t,U,f'Temperature variation in the Kangerlussuaq soil during {nb_years} years {complementary_title}',units=['m','year'],cmap=colormap,vmin=vmin,vmax=vmax)

        #Finding the temperature array in the ground after nb_years in winter and summer
        
        loc = int(-S_IN_YEAR/dt) # Final year of the result.

        # Extract the min values over the final year=winter
        winter = U[:, loc:].min(axis=1)
        # Extract the max values over the final year=summer
        summer = U[:, loc:].max(axis=1)
        #Finding the depth at which permafrost is isothermal
        for k in range(1,len(winter)):
            if (summer[-k]-winter[-k]<0.1):
                isothermal_depth = x[-k]
                break
        #Finding the depth at which the permafrost base
        for k in range(len(winter)):
            if (winter[k]<0):
                permafrost_base_depth = x[k]
                break
        #Finding the depth of the active layer
        for k in range(1,len(winter)):
            if (summer[-k]<0):
                active_layer_depth = x[-k]
                break

        #Plotting the temperature profile in the ground in summer and winter after nb_years
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

        #plotting winter and summer temperature profiles
        ax2.plot(winter, x, label='Winter')
        ax2.plot(summer, x, label='Summer')

        #Plotting permafrost layers

            #Isothermal layer
        ax2.axhline(isothermal_depth, color='k', linestyle='--')
        ax2.text(0,isothermal_depth+3,f'Isothermal permafrost', color='k')
        ax2.text(plt.xlim()[1] + 0.5,isothermal_depth,f'{isothermal_depth:.1f} m', color='k')

            #Permafrost base
        ax2.axhline(permafrost_base_depth, color='b', linestyle='--')
        ax2.text(0,permafrost_base_depth+3,f'Permafrost base', color='b')
        ax2.text(plt.xlim()[1] + 0.5,permafrost_base_depth,f'{permafrost_base_depth:.1f} m', color='b')
            
            #Active layer
        ax2.axhline(active_layer_depth, color='g', linestyle='--')
        ax2.text(0,active_layer_depth+3,f'Active layer', color='g')
        ax2.text(plt.xlim()[1] + 0.5,active_layer_depth,f'{active_layer_depth:.1f} m', color='g')

        #Plotting the permafrost thickness
        permafrost_thickness = active_layer_depth - permafrost_base_depth
        ax2.annotate('', xy=(-10, permafrost_base_depth), xytext=(-10, active_layer_depth), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax2.text(-12, permafrost_base_depth + 5, f'Permafrost thickness: {permafrost_thickness:.1f} m', color='red', rotation=90)

        #Adding axes label and title
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Depth (m)')
        ax2.set_title(f'Soil Temperature Profile in Kangerlussuaq after {nb_years} years {complementary_title}')
        ax2.legend()

    return x,t,U

    
def plot_heatsolve(x,t,U,title,units=['m','s'],**kwargs):
    '''
    Plot the 2D solution for the 'solve_heat' function

    extra kwargs handed for color

    Parameters:
    ------------
    x,t : 1D Numpy arrays
        Space and time values, respectively
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    title: string
        the title for the plot
    
    Returns:
    ----------
    fig,ax : Matplotlib figure and axis object:
        the figure and axes of the plot
    
    cbar : Matplotlib color bar object
        the color bar on the final plot

    '''

    #Create and configure a figure & axes:
    fig,ax=plt.subplots(1,1,figsize=(8,7))

    #check out for kwargs default
    if 'cmap' not in kwargs:
        kwargs['cmap']='hot'

    #Add contour to our axes
    contour=ax.pcolormesh(t,x,U,shading='auto',**kwargs)
    cbar=plt.colorbar(contour)

    #Add labels to the plot
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel(f'Time (${units[1]}$)')
    ax.set_ylabel(f'Position (${units[0]}$)')
    ax.set_title(title)
    plt.show()

    return fig,ax,cbar



'''
c2=0.25e-6 #m2/s
x_init=0 #m
x_final=100 #m
dx= #m
t_init=0 #s
t_final= #s
dt= #s
lowerbound=5 #°C
upperbound=temp_kanger #callable
'''
