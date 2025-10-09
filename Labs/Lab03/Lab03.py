#!/usr/bin/env python3
'''
Author: Adrien Boissy
Collaborators: None

'''

import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('seaborn-darkgrid')

def solve_heat(c2=1,x_init=0,x_final=1,dx=0.2,t_init=0,t_final=0.2,dt=0.02,T_border=0):
    """
    Parameters:
    ----------
    c2: float
        c^2, the square of the diffusion coefficient
     XXXXX

     XXXXX
     XXXX


    returns:
    ---------
    x,t : 1D Numpy arrays
        Space and time values, respectively
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime

    """

    

    N=int((t_final-t_init)/dt)
    M=int((x_final-x_init)/dx)
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
    fig,ax=plt.subplots(1,1,figsize=(8,8))

    #check out for kwargs default
    if 'cmap' not in kwargs:
        kwargs['cmap']='hot'

    #Add contour to our axes
    contour=ax.pcolor(t,x,U,**kwargs)
    cbar=plt.colorbar(contour)

    #Add labels to the plot
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Position ($m$)')
    ax.set_title(title)

    return fig,ax,cbar