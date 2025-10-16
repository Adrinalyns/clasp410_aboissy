#!/usr/bin/env python3
'''
Author: Adrien Boissy
Collaborators: None

'''

import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('seaborn-darkgrid')

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
    lowerbound : float or function or None, default 0
        The Dirichlet boundary condition at x_init if not None (otherwise Neumann BC with a zero gradient)
    upperbound : float or function or None, default 0
        The Dirichlet boundary condition at x_final if not None (otherwise Neumann BC with a zero gradient)

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
            U[0,j+1] = lowerbound(t[j+1]) #Dirichlet boundary condition function of time
        else:
            U[0,j+1] = lowerbound #Dirichlet constant boundary condition
    
        if upperbound is None:
            U[-1,j+1] = U[-2,j+1] #Neumann boundary condition at x=x_final
        elif callable(upperbound):
            U[-1,j+1] = upperbound(t[j+1]) #Dirichlet boundary condition function of time
        else:
            U[-1,j+1] = upperbound #Dirichlet constant boundary condition
    return t,x,U

def test_solve_heat():
    '''
    This function tests the solve_heat function against a known solution.
    The parameters are :
    c2=1 m^2/s
    x_init=0 m
    x_final=1 m
    dx=0.2 m
    t_init=0 s
    t_final=0.2 s
    dt=0.02 s
    U(x,0)= 4x-4x^2
    lowerbound = upperbound = 0 °C
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
            f'\nthe difference between the computed solution and the known solution is too high : \n error of {abs(U[i,j]-sol10p3[i,j]):.5f}' + \
            f' higher than {max_error_allowed}'

            #Change the maximum error found if the current error is higher
            if abs(U[i,j]-sol10p3[i,j])>max_error:
                max_error=abs(U[i,j]-sol10p3[i,j])

    print(f'Test passed ! \n the calculated solution is close to the known solution with a maximum error of {max_error:.5f} (max allowed is {max_error_allowed})')


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