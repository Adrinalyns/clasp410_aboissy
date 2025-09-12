#!/usr/bin/env python3
'''
This code solves the N-layer atmosphere model, where each layer as an uniform emissivity epsilon

'''
import numpy as np
import matplotlib.pyplot as plt

sigma=5.6703e-8

def solve_N_layer(n_layers,epsilon,s0=1370.,alpha=0.3):
    '''
    This code solves the N-layer atmosphere model (with a constant emissivity)
    by resolving the matrix equation AX=b.
    The matrix equation is only a way to unite all the N+1 equations 
    (energy balance for the N layers and the energy balance of Earth's surface)

    Parameters:
    -------------

    n_layers: Integer
        The number of layer of atmosphere in the model

    s0: float, default 1370W/m2
        The solar constant
    
    alpha: float, default to 0.3
        The Earth albedo

    epsilon: float
        the emissivity of the layers (uniform in this model)

    Returns:
    ----------
    t_array: vector of float
        The temperature of Earth's surface in Celsius followed by 
        the Temperature of each layer in Celsius

    '''
    #introducing a debug variable
    debug=False

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([n_layers+1, n_layers+1])
    b = np.zeros(n_layers+1)


    # Populate based on our model:
    # the first equation is different from the other one, let populate A and b for this equation
    b[0]=-s0*(1-alpha)/4*epsilon
    A[0,0]=-epsilon
    for j in range(1,n_layers+1):
        A[0,j]=epsilon*(1-epsilon)**(j-1)
    
    #the remaining N equations should be completed with the generic formula that we found
    for i in range(1,n_layers+1):
        for j in range(n_layers+1):
            if(j>=0 and j<i):
                A[i, j] =epsilon*(1-epsilon)**((i-1)-j)
            elif(j>i and j<=n_layers):
                A[i, j] =epsilon*(1-epsilon)**(j-(i+1))
            else:
                A[i,j]=-2
        b[i] =0
    
    if debug:
        print(A,b)

    # Invert matrix:
    Ainv = np.linalg.inv(A)
    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!
    
    #The earth is considered as a Black body so its emissivity is 1 
    #and can be different from epsilon, so a different calculus is needed 
    #to derive Erath Temperature from its radiation flux, than for the atmosphere layers
    t0=(fluxes[0]/sigma)**0.25 
    t_array=(fluxes/sigma/epsilon)**0.25
    t_array[0]=t0

    return t_array


def validate_model(detail):
    '''
    This function compare the temperature array obtained from our model
    to the Temperature obtained from this website :
    https://singh.sci.monash.edu/models/Nlayer/N_layer.html

    parameters:
    ----------
    detail: boolean
        You can chose detail=0 if you only want see the maximum difference between our code and the ideal one
        You can chose detail=1 if you want to see all the detail of layers'temperature

    We will take 2 examples :
    - n_layers=4, epsilon=1, alpha=0.3, s0=1370 W/m2
    t_real1=[381.3,360.6,335.6,303.3,255.0]
    - n_layers=6, epsilon=0.45, alpha=0.33, s0=1350 W/m2
    t_real2=[323.4,302.4,291.3,278.9,262.2,247.1,225.2]
    '''

    t_real1=[381.3,360.6,335.6,303.3,255.0]
    t_code1=solve_N_layer(4,1)

    print(f"Validation 1 : n_layers=4, epsilon=1, alpha=0.3, s0=1370 W/m2 \n")

    if(detail):

        print(f'\tThe coded temperature of Earth is {t_code1[0]}')
        print(f'\tThe real temperature of Earth is {t_real1[0]}')
        for i in range(1,4):
            print(f'\tThe coded temperature of the layer {i} is {t_code1[i]} °C')
            print(f'\tThe real temperature of the layer {i} is {t_real1[i]} 0C')
    
    #Calculation of the max difference between our model and the theoritical one
    max_error=max(t_real1-t_code1)
    
    print(f'\n \tThe maximum difference in temperature between our model and the ideal one is {max_error} °C')

    print("\nValidation 2 : n_layers=6, epsilon=0.45, alpha=0.33, s0=1350 W/m2 \n")
    t_real2=[323.4,302.4,291.3,278.9,262.2,247.1,225.2]
    t_code2=solve_N_layer(6,0.45,s0=1350,alpha=0.33)

    if(detail):
        print(f'\tThe coded temperature of Earth is {t_code2[0]}')
        print(f'\tThe real temperature of Earth is {t_real2[0]}')
        for i in range(1,6):
            print(f'\tThe coded temperature of the layer {i} is {t_code2[i]} °C')
            print(f'\tThe real temperature of the layer {i} is {t_real2[i]} 0C')
        
    #Calculation of the max difference between our model and the theoritical one
    max_error=max(t_real2-t_code2)
    
    print(f'\n \tThe maximum difference in temperature between our model and the ideal one is {max_error} °C')