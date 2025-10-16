#!/usr/bin/env python3
'''
Author: Adrien Boissy
Collaborators: None

'''

import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('seaborn-darkgrid')

def solve_heat_Dirichlet(c2=1,x_init=0,x_final=1,dx=0.02,t_init=0,t_final=0.2,dt=0.0002,T_border=0):
    """
    Solve the Heat equation for a bar with Dirichlet boundary conditions, and an initial condition of Ti(x)=4x-4x^2
    
    Parameters:
    ----------
    c2: float
        c^2, the square of the diffusion coefficient
    x_init, x_final : float
        The initial and final space values, they are included in the grid
    dx : float
        The space step
    t_init, t_final : float
        The initial and final time values, they are included in the grid
    dt : float
        The time step
    T_border : float, default 0
        The temperature at the borders of the grid

    returns:
    ---------
    x,t : 1D Numpy arrays
        Space and time values, respectively
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime

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
    U[:,0]=4*x -4*x**2

    #Get the r coeff
    r=c2*(dt/dx**2)
    #Solve equation
    for j in range(N-1):
        U[1:M-1,j+1] = (1-2*r)*U[1:M-1,j]+ r*(U[2:M,j]+ U[:M-2,j])

    return t,x,U

def solve_heat(c2=1,x_init=0,x_final=1,dx=0.02,t_init=0,t_final=0.2,dt=0.0002,lowerbound=0,upperbound=0):
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
    t_init, t_final : float
        The initial and final time values, they are included in the grid
    dt : float
        The time step
    T_border : float, default 0
        The temperature at the borders of the grid

    returns:
    ---------
    x,t : 1D Numpy arrays
        Space and time values, respectively
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime

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
    U[:,0]=4*x -4*x**2

    #Get the r coeff
    r=c2*(dt/dx**2)
    #Solve equation
    for j in range(N-1):
        U[1:M-1,j+1] = (1-2*r)*U[1:M-1,j]+ r*(U[2:M,j]+ U[:M-2,j])
        if lowerbound is None:
            U[0,j+1] = U[1,j+1] #Neumann boundary condition at x=x_init
        elif callable(lowerbound):
            U[0,j+1] = lowerbound(t[j+1])
        else:
            U[0,j+1] = lowerbound
    
        if upperbound is None:
            U[-1,j+1] = U[-2,j+1] #Neumann boundary condition at x=x_final
        elif callable(upperbound):
            U[-1,j+1] = upperbound(t[j+1])
        else:
            U[-1,j+1] = upperbound
    return t,x,U

def plot_heatsolve(x,t,U,title,**kwargs):
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
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Position ($m$)')
    ax.set_title(title)

    return fig,ax,cbar