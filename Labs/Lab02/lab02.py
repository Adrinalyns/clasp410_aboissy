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
        dN1, dN2 = func(time[i-1], [N1[i-1], N2[i-1]], **kwargs )
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
        Largest time step allowed in years.

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


def validate_model():
    """
    This function validate the Euler, RK8 solver, and the functions that calculates the derivatives
    by plotting the results obtained with the same parameters as in figure 1 (the example given in the HW)
    """
    
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

    time_comp,N1_comp,N2_comp=euler_solve(dNdt_comp, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final,a=a,b=b,c=c,d=d)
    
    ax1.plot(time_comp,N1_comp,label=f'N1 with Euler')
    ax1.plot(time_comp,N2_comp,label=f'N2 with Euler')
    
    #RK8 method

    time_rk8_comp,N1_rk8_comp,N2_rk8_comp=solve_rk8(dNdt_comp, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final,a=a,b=b,c=c,d=d)

    ax1.plot(time_rk8_comp,N1_rk8_comp, label=f'N1 with RK8',linestyle='--')
    ax1.plot(time_rk8_comp,N2_rk8_comp, label=f'N2 with RK8',linestyle='--')
    ax1.set_title("Lokta Volterra Competition model")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel(r'$\frac{Population}{carrying-cap}$')
    ax1.legend()

    #prey-predator model
    dt=0.05
        #Euler method

    time_PP,N1_PP,N2_PP=euler_solve(dNdt_predator_prey, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final,a=a,b=b,c=c,d=d)

    ax2.plot(time_PP,N1_PP,label=f'N1 (Prey) with Euler')
    ax2.plot(time_PP,N2_PP,label=f'N2 (Predator) with Euler')

        #RK8 method

    time_rk8_PP,N1_rk8_PP,N2_rk8_PP=solve_rk8(dNdt_predator_prey, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final,a=a,b=b,c=c,d=d)

    ax2.plot(time_rk8_PP,N1_rk8_PP,label=f'N1 (Prey) with RK8',linestyle='--')
    ax2.plot(time_rk8_PP,N2_rk8_PP, label=f'N2 (Predator) with RK8',linestyle='--')
    ax2.set_title("Lokta Volterra Predator-Prey model")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel(r'$\frac{Population}{carrying-cap}$')
    ax2.legend()
    
def question1():
    """
    This function answers question 1 by plotting the results of both solvers with different step times.
    We use the same situation as in Figure 1, and we change the step time, to see to what extent it 
    influences the results for both methods.
    """
    a=1
    b=2
    c=1
    d=3
    N1_init=0.3
    N2_init=0.6
    t_final=100

    
    #competition model
    dt_array=[0.05,0.5,1,2]
    for dt in dt_array:
        
        #Calculating the results for the 2 Methods

            #Euler method
        time_comp,N1_comp,N2_comp=euler_solve(dNdt_comp, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final,a=a,b=b,c=c,d=d)
            #RK8 method
        time_rk8_comp,N1_rk8_comp,N2_rk8_comp=solve_rk8(dNdt_comp, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final,a=a,b=b,c=c,d=d)

        #print(f'For dt = {dt} years, the Euler solution took {time_comp.size} iterations.')
        #print(f'For dt = {dt} years, the RK8 solution took {time_rk8_comp.size} iterations.')

        #plotting figures
        plt.figure()    #create a new figure for next plot

        plt.plot(time_comp,N1_comp,label=f'N1 with Euler')
        plt.plot(time_comp,N2_comp,label=f'N2 with Euler')

        plt.plot(time_rk8_comp,N1_rk8_comp, label=f'N1 with RK8',linestyle='--')
        plt.plot(time_rk8_comp,N2_rk8_comp, label=f'N2 with RK8',linestyle='--')

        plt.title(f"Lokta Volterra Competition model dt = {dt} year")
        plt.xlabel("Time (years)")
        plt.ylabel(r'$\frac{Population}{carrying-cap}$')
        plt.legend()
        
    
    #prey-predator model

    dt_array=[0.001,0.01,0.05,0.09]
    for dt in dt_array:

        #Calculating the results for the 2 Methods

            #Euler method
        time_PP,N1_PP,N2_PP=euler_solve(dNdt_predator_prey, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final,a=a,b=b,c=c,d=d)
            
            #RK8 method
        time_rk8_PP,N1_rk8_PP,N2_rk8_PP=solve_rk8(dNdt_predator_prey, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final,a=a,b=b,c=c,d=d)

        #Plotting figures

        plt.figure() #create a new figure for next plot

        plt.plot(time_PP,N1_PP,label=f'N1 (Prey) with Euler')
        plt.plot(time_PP,N2_PP,label=f'N2 (Predator) with Euler')
            
        plt.plot(time_rk8_PP,N1_rk8_PP,label=f'N1 (Prey) with RK8',linestyle='--')
        plt.plot(time_rk8_PP,N2_rk8_PP, label=f'N2 (Predator) with RK8',linestyle='--')

        plt.title(f"Lokta Volterra Predator-Prey model dt = {dt} year")
        plt.xlabel("Time (years)")
        plt.ylabel(r'$\frac{Population}{carrying-cap}$')
        plt.legend()
    
def question2():
    """
    This function plots all the equilibrium states I have found. 
    Thus it allows to answer question2, and highlights the influence 
    of initial conditions and parameters on the behavior of both populations
    """
    
    dt=1.
    t_final=100.

    #this array contains every equilibrium state that I found.
    # each equilibrium is represented by an array containing the values of:
    # [a,b,c,d,N1_init,N2_init]
    equilibriums=[[1,1,1,1,0.2,0.8],[1,1,1,1,0.8,0.8],[1,1,1,1,0.2,0.2],[1,1,1,1,0.2,0.5],[1,1,1,1,0.8,0.5],[1,2,1,5,0.15,0.6],[1,2,1,7,0.1,0.6],[1,3,1,2,0.6,0.3],[1,2,1,3,0.3,0.6]]

    for equilibrium in equilibriums:
        #Settings for each equilibrium
        a=equilibrium[0]
        b=equilibrium[1]
        c=equilibrium[2]
        d=equilibrium[3]
        N1_init=equilibrium[4]
        N2_init=equilibrium[5]

        #Calculating each equilibrium
        time,N1,N2=solve_rk8(dNdt_comp, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=t_final,a=a,b=b,c=c,d=d)

        #Plotting each equilibrium
        plt.figure()    #create a new figure for next plot
        plt.plot(time,N1, label=f'N1 with RK8',linestyle='--')
        plt.plot(time,N2, label=f'N2 with RK8',linestyle='--')
        plt.title(f"Lokta Volterra Competition model\n dt={dt} year a={a}, b={b}, c={c}, d={d}, N1(0)={N1_init}, N2(0)={N2_init}")
        plt.xlabel("Time (years)")
        plt.ylabel(r'$\frac{Population}{carrying-cap}$')
        plt.legend()

dt=0.1
a=1
b=2
c=1
d=3
N1_init=0.5
N2_init=0.5
t_final=100

for a in range(1,4):
    for b in range(1,4):
        for c in range(1,4):
            for d in range(1,4):

                #Calculating each equilibrium
                time,N1,N2=solve_rk8(dNdt_predator_prey, N1_init=N1_init/10, N2_init=N2_init/10, dt=dt, t_final=t_final,a=a,b=b,c=c,d=d)
                #Plotting phase diagram
                #plt.figure()
                plt.plot(N1,N2,label=f'a={a}, b={b}, c={c}, d={d}, N1(0)={N1_init}, N2(0)={N2_init}')
                plt.title(f'Phase diagram with \n dt={dt} year a={a}, b={b}, c={c}, d={d}, N1(0)={N1_init}, N2(0)={N2_init}')
                plt.xlabel(r'N1-Prey ($\frac{Population}{carrying-cap}$)')
                plt.ylabel(r'N2-Predators ($\frac{Population}{carrying-cap}$)')
                plt.legend()
        """
        #Plotting each equilibrium
        plt.figure()    #create a new figure for next plot
        plt.plot(time,N1, label=f'N1 with RK8',linestyle='--')
        plt.plot(time,N2, label=f'N2 with RK8',linestyle='--')
        plt.title(f"Lokta Volterra Competition model\n dt={dt} year a={a}, b={b}, c={c}, d={d}, N1(0)={N1_init}, N2(0)={N2_init}")
        plt.xlabel("Time (years)")
        plt.ylabel(r'$\frac{Population}{carrying-cap}$')
        plt.legend()
        """