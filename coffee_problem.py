#!/usr/bin/env python3
'''
This code solves the coffee problem to learn when to put the cream in the coffee

'''
import numpy as np

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

