#!/usr/bin/env python3
'''
This code solves the coffee problem to learn when to put the cream in the coffee

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.style.use('seaborn-darkgrid')

def solve_temp(t, T_init=90, T_env=20.0, k=1/300.):
    '''
    This function returns Temperature as a function of time using Newton's law of cooling

    Parameters
    -----------
    t: Numpy array
        an array of time values in seconds

    T_init: floating point, default to 90
        Initial temperature in celsius
        
    T_env: floating point, default to 90
        Initial temperature in celsius

    k: floating point, default 1/300.
        the heat transfer coefficient
    Returns
    ----------
    t_coffee: Numpy array
        Temperature corresponding to time t
    '''
    T_coffee= T_env + (T_init - T_env)*np.exp(-k*t)

    return T_coffee

def time_to_temp(T_final, T_env=20.0, T_init=90., k=1/300.):
    '''
    Considering a Target temperature, function calculate the time needed for the coffee to cool from an initial temperature


    Parameters
    -----------

    T_final: floating point, 
        Final Temperature in celsius

    T_init: floating point, default to 90
        Initial temperature in celsius
        
    T_env: floating point, default to 90
        Initial temperature in celsius

    k: floating point, default 1/300.
        the heat transfer coefficient
    Returns
    ----------
    t: float
        the time needed for the coffee to cool from T_init to T_final

    '''
    t=(-1/k)*np.log((T_final-T_env)/(T_init-T_env))
    return t

def verify_code():
    '''
    Verify that our implementation is correct
    using example problem from :
    URL
    '''
    t_real=60*10.76
    k=np.log(95/110.)/-120
    t_code=time_to_temp(120,T_init=180,T_env=70,k=k)
    print("Target solution is: ",t_real)
    print("Numerical solution is: ",t_code)
    print("The difference is:", t_real-t_code)

def euler_coffee(dt=.25,k=1/300,T_env=20.0, T_init=90., t_final=300.):
    '''
    Solve the cooling equation with Euler method

    Parameters
    -----------
    dt: float, default to 0.25
        the step time for resolving the equation 

    T_init: floating point, default to 90
        Initial temperature in celsius
        
    T_env: floating point, default to 20
        Initial temperature in celsius

    k: floating point, default 1/300.
        the heat transfer coefficient
    
    t_final: floating point, default to 300
        number of seconds of the simulation

    Returns
    ----------
    t_coffee: Numpy array
        Temperature corresponding to time t
    
    '''
    time=np.arange(0,t_final,dt)
    temp=np.zeros(time.size)
    temp[0]=T_init

    #Solve
    for i in range(time.size-1):
        temp[i+1]=temp[i]-dt*k*(temp[i]-T_env)
    
    return time,temp

def solve_euler(dfx,dt=.25,f0=90.,t_start=0.,t_final=300.,**kwargs):
    """
    Solve an ordinary differential equation using Euler's method
    extra kwargs are passed to the dfx function

    Parameters:
    -----------
    dfx: function
        A function representing the time derivative of our diffyQ. It should
        take 2 argument: the current time, the current value of the function
        and return 1 value: the derivative at time 't'

    f0: float
        initial condition for the differential equation
    
    t_start, t_final: float, default to 0,300 respectively
        the starting time and ending time
    
    dt: float
        the step time for the resolution

    Returns:
    ----------

    t: Numpy array
        Time in second over the entire solution
    fx: Numpy array
        the solution as a function of time
    
    """
    #configure our problem:
    time= np.arange(t_start,t_final,dt)
    fx=np.zeros(time.size)
    fx[0]=f0
    #Solve

    for i in range(time.size-1):
        fx[i+1]=fx[i]+dt*dfx(time[i],fx[i],**kwargs)

    return time,fx

def solve_rk8(dfx,dt=5,f0=90.,t_start=0.,t_final=300.,**kwargs):
    """
    Solve an ordinary differential equation using Euler's method
    extra kwargs are passed to the dfx function

    Parameters:
    -----------
    dfx: function
        A function representing the time derivative of our diffyQ. It should
        take 2 argument: the current time, the current value of the function
        and return 1 value: the derivative at time 't'

    f0: float
        initial condition for the differential equation
    
    t_start, t_final: float, default to 0,300 respectively
        the starting time and ending time
    
    dt: float
        the step time for the resolution

    Returns:
    ----------

    t: Numpy array
        Time in second over the entire solution
    fx: Numpy array
        the solution as a function of time
    
    """
    #Solve

    result=solve_ivp(dfx, [t_start,t_final], [f0],method='DOP853')

    return result.t, result.y[0,:]

def newtcool(t,Tnow,k=1/300.,T_env=20.0):
    '''
    Newton's law of cooling: given time t, Temperature now (Tnow), a cooling
    coefficient (k), and an environmental temp (T_env), return the rate of cooling
    (i.E., dT/dt)
    '''
    return -k*(Tnow-T_env)


def newtcool(t,Tnow,k=1/300.,T_env=20.0):
    '''
    Newton's law of cooling: given time t, Temperature now (Tnow), a cooling
    coefficient (k), and an environmental temp (T_env), return the rate of cooling
    (i.E., dT/dt)
    '''
    return -k*(Tnow-T_env)


def solve_coffee_pb():
    #Solve the actual problem using the functions declared above

    #Quantitatively
    t1=time_to_temp(65)         #add cream at the end, when the coffee is 65°C
    t2=time_to_temp(60,T_init=85)#add cream at the beginning, when the coffee is 90°C
    tc=time_to_temp(60)         #Control case : no cream

    print(f"TIME TO DRINKABLE COFFEE:")
    print(f"\tControl case = {tc:.2f}s")
    print(f"\tAdd cream later = {t1:.2f}s")
    print(f"\tAdd cream before = {t2:.2f}s")

    #create time series of Temperatures for cooling coffee
    t=np.arange(0,600,0.5)
    temp1= solve_temp(t)
    temp2= solve_temp(t, T_init=85.)

    

    #Create our figure and plot stuff
    fig, ax = plt.subplots(1,1)
    ax.plot(t,temp1, label=f'Add Cream Later (T={t1:.1f}s)')
    ax.plot(t,temp2, label=f'Add Cream Before (T={t2:.1f}s)')
    

    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('When to add cream: Getting a coffee cooled quickly')


    fig2,(ax1,ax2)=plt.subplots(2,1)
    ax1.plot(t,temp1)
    ax2.plot(t,temp2)

def numerical_methods():

    #Euler Method : create time series of Temperatures for cooling coffee 

    t=np.arange(0,600,0.5)
    temp1= solve_temp(t)
    dt=.25
    etime,etemp=euler_coffee(dt=dt,t_final=600)
    etime2,etemp2=solve_euler(newtcool, T_env=0.)
    time3,temp3=solve_rk8(newtcool)

    fig, ax = plt.subplots(1,1,figsize=[6.4, 4.8])
    ax.plot(t,temp1, label=f'Numerical solution')
    ax.plot(etime2,etemp2, label=f'Euler numerical solution for dt={dt}s')
    ax.plot(time3,temp3, label=f'RK8 solution')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Analytical vs Numerical')


