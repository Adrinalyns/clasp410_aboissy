#!/usr/bin/env python3
'''
Author: Adrien Boissy
Collaborators: None

'''

def solve_heat(c2=1,x_init=0,x_final=0.1,dx=0.2,t_init,t_final=0.2,dt=0.02,T_border):
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

    

    N=(t_final-t_init)//dt
    M=(x_final-x_init)/dx
    #Set up space and time grid
    t = np.linspace(t_init,t_final,N)
    x = np.linspace(x_init,x_final,N)
    #Create solution matrix and set initial condition
    U=np.zeros(M,N)
    U[:,0]=4*x -4*x**2

    #Get the r coeff
    r=c2*(dt/dx**2)
    #Solve equation

    for j in range(N-1):
        U[1:M-1,j+1] = (1-2*r)*U[1:M-1,j]+ r*(U[2:M,j]+ U[:M-2,j])

    
