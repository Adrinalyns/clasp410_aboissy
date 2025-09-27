#!/usr/bin/env python3
'''
This code solves the Lokta-Voterra equations, for:
- a predator-prey situation
- two species in competition

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:

XXXXXXXXXXXXX

'''


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function uses the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    Parameters
    ----------
    t : float
        The current time (not used here).

    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).

    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]

    return dN1dt, dN2dt

def dNdt_predator_prey(t, N, a=1, b=2, c=1, d=3):
    '''
    This function uses the Lotka-Volterra predator-prey equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    Parameters
    ----------
    t : float
        The current time (not used here).

    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).

    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[1]*N[0]

    return dN1dt, dN2dt

def euler_solve(func, N1_init, N2_init, dt, t_final=100.0, **kwargs):
    '''
    This function uses Euler method to solve an ordinary differential 
    equation over a period of time.

    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.

    N1_init : float
        The initial value of N1 (at time=0)

    N2_init : float
        The initial value of N2 (at time=0)

    dt : float
        The time step for the resolution in years
    
    t_final : float, default to 100.0
        The ending time of the resolution in years
    
    '''
    time=np.arange(0,100.0,dt)

    N1=np.zeros(time.size)
    N1[0]=N1_init

    N2=np.zeros(time.size)
    N2[0]=N2_init

    for i in range(1, time.size):
        dN1, dN2 = func(i, [N1[i-1], N2[i-1]], **kwargs )
        N1[i]= N1[i-1] + dt*dN1
        N2[i]= N2[i-1] + dt*dN2

    return time,N1,N2

def solve_rk8(func, N1_init=.5, N2_init=.5, dt=10, t_final=100.0, a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.

    Parameters
    ----------

    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.

    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]

    dt : float, default=10
        Largest timestep allowed in years.

    t_final : float, default=100
        Integrate until this value is reached, in years.

    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values

    Returns
    -------
    time : Numpy array
        Time elapsed in years.

    N1, N2 : Numpy arrays
        Normalized population density solutions.
    '''
    from scipy.integrate import solve_ivp
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
    args=[a, b, c, d], method='DOP853', max_step=dt)
    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]
    # Return values to caller.
    return time, N1, N2


def question1():
    a=1
    b=2
    c=1
    d=3
    N1_init=0.3
    N2_init=0.6
    t_final=100

    fig,(ax1,ax2)=plt.subplots(1,2)

    #competition model
    dt=1
        #Euler method

    time,N1,N2=euler_solve(dNdt_comp, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final)

    ax1.plot(time,N1)
    ax1.plot(time,N2)

    #prey-predator model
    dt=0.05
        #Euler method

    time,N1,N2=euler_solve(dNdt_predator_prey, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final)

    fig,ax=plt.subplots(1,1)

    ax2.plot(time,N1)
    ax2.plot(time,N2)
    
    