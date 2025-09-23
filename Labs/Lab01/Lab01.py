#!/usr/bin/env python3
'''
This code solves the N-layer atmosphere model, where each layer as an uniform emissivity epsilon.
It plots all the results used in the report

TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:

- validate_model(0) : to validate the N layer model quickly
- validate_model(1) : to validate the N layer model in detail (use it to debug the model)

- question3a() : to plot the graph for the first part of question 3) : temperature wrt emissivity
- question3b() : to plot the graph for the second part of question 3): temperature wrt number of layers
- question4()  : to plot the graph for question 4): venus number layer
- question5()  : to plot the graph for question 5): nuclear winter


'''
import numpy as np
import matplotlib.pyplot as plt
import time

plt.style.use('seaborn-darkgrid')

#Physical constant
sigma=5.6703e-8 # W/m2/K-4
t_earth_surface=288. # K

def solve_N_layer(n_layers,epsilon,s0=1370.,alpha=0.33,debug=0):
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
    
    alpha: float, default to 0.33
        The Earth albedo

    epsilon: float
        the emissivity of the layers (uniform in this model)

    debug: bool, default to 0
        When debug is set to 1, it prints out the matrix A and the array b

    Returns:
    ----------
    t_array: vector of float
        The temperature of Earth's surface in Celsius followed by 
        the Temperature of each layer in Celsius

    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([n_layers+1, n_layers+1])
    b = np.zeros(n_layers+1)

    # Populate based on our model:
    # the first equation is different from the other one, however I multiplied this equation by epsilon
    # to have truly symmetrical matrix, this is why b[0] is multiplied by epsilon
    b[0]=-s0*(1-alpha)/4*epsilon
    
    #We can now populate the matrix, to make it faster, I first populate it like the diagonal was full of -2,
    for i in range(n_layers+1):
        for j in range(n_layers+1):
            if i==j:
                A[i,j]=-2
            else:
                A[i,j]=epsilon*(1-epsilon)**(abs(j-i)-1) # I figured out that using np.abs() make it much slower than using abs()
    
    #Then I change A[0,0] number to respect the real matrix, it should be -1 but because 
    #I multiplied the first equation by epsilon it becomes -epsilon
    A[0,0]=-epsilon 
    
    """
    or : if i==j:
            A[i,j]=-2+1*(j==0)
        else:
            A[i,j]=epsilon**(i>0)*(1-epsilon)**(np.abs(j-i)-1)
    """

    if debug:
        print(A,b)
    # Invert matrix:
    Ainv = np.linalg.inv(A)
    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!
    
    #The earth is considered as a Black body so its emissivity is 1 
    #and can be different from epsilon, so a different calculus is needed 
    #to derive Earth Temperature from its radiation flux, than for the atmosphere layers
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

    returns:
    -------
    It prints the difference between the results given by my code and the wanted results in the following 2 situations

    We will take 2 examples :
    - n_layers=4, epsilon=1, alpha=0.3, s0=1370 W/m2
      t_real1=[381.3,360.6,335.6,303.3,255.0]
    - n_layers=6, epsilon=0.45, alpha=0.33, s0=1350 W/m2
      t_real2=[323.4,302.4,291.3,278.9,262.2,247.1,225.2]
    '''

    t_real1=[381.3,360.6,335.6,303.3,255.0]
    t_code1=solve_N_layer(4,1,alpha=0.3)

    print(f"Validation 1 : n_layers=4, epsilon=1, alpha=0.3, s0=1370 W/m2 \n")

    if(detail):

        print(f'\tThe coded temperature of Earth is {t_code1[0]}')
        print(f'\tThe real temperature of Earth is {t_real1[0]}')
        for i in range(1,4):
            print(f'\tThe coded temperature of the layer {i} is {t_code1[i]} °C')
            print(f'\tThe real temperature of the layer {i} is {t_real1[i]} °C')
    
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
            print(f'\tThe real temperature of the layer {i} is {t_real2[i]} °C')
        
    #Calculation of the max difference between our model and the theoritical one
    max_error=max(t_real2-t_code2)
    
    print(f'\n \tThe maximum difference in temperature between our model and the ideal one is {max_error} °C')

def solve_N_layer_nuclear_winter(n_layers,epsilon,alpha,s0=1350.,debug=0):
    '''
    This code solves the N-layer atmosphere model (with a constant emissivity)
    in case of a nuclear winter–the solar radiation is absorbed by the top layer of the atmosphere 
    instead of being absorbed by the ground–by resolving the matrix equation AX=b.
    The matrix equation is only a way to unite all the N+1 equations 
    (energy balance for the N layers and the energy balance of Earth's surface)

    Parameters:
    -------------

    n_layers: Integer
        The number of layer of atmosphere in the model

    s0: float, default 1370W/m2
        The solar constant
    
    alpha: float, default to 1
        The Earth albedo

    epsilon: float
        the emissivity of the layers (uniform in this model)

    debug: bool, default to 0
        When debug is set to 1, it prints out the matrix A and the array b

    Returns:
    ----------
    t_array: vector of float
        The temperature of Earth's surface in Celsius followed by 
        the Temperature of each layer in Celsius

    '''
    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([n_layers+1, n_layers+1])
    b = np.zeros(n_layers+1)

    # Populate based on our model:
    # the first equation is different from the other one, however I multiplied this equation by epsilon
    # to have truly symmetrical matrix, this doesn't change anything in that case : b[0]=0*epsilon=0 
    #And b[N]= solar flux, but because the first layer is opaque, non of the solar flux is reflected : alpha=0
    b[n_layers]=-s0*(1-alpha)/4
    
    #We can now populate the matrix, to make it faster, I first populate it like the diagonal was full of -2,
    for i in range(n_layers+1):
        for j in range(n_layers+1):
            if debug:
                print(f'A[i={i},j={j}] = {A[i, j]}')
            if i==j:
                A[i,j]=-2
            else:
                A[i,j]=epsilon*(1-epsilon)**(abs(j-i)-1) # I figured out that using np.abs() make it much slower than using abs()
    
    #Then I change A[0,0] number to respect the real matrix, it should be -1 but because 
    #I multiplied the first equation by epsilon it becomes -epsilon
    A[0,0]=-epsilon 
    
    """
    or : if i==j:
            A[i,j]=-2+1*(j==0)
        else:
            A[i,j]=epsilon**(i>0)*(1-epsilon)**(np.abs(j-i)-1)
    """

    if debug:
        print(A,b)
    # Invert matrix:
    Ainv = np.linalg.inv(A)
    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!
    
    #The earth is considered as a Black body so its emissivity is 1 
    #and can be different from epsilon, so a different calculus is needed 
    #to derive Earth Temperature from its radiation flux, than for the atmosphere layers
    t0=(fluxes[0]/sigma)**0.25 
    t_array=(fluxes/sigma/epsilon)**0.25
    t_array[0]=t0

    return t_array

def question3a():
    """
    This function answers the first part of question 3) by plotting the Earth's surface temperature in a single-layer atmosphere
    for emissivity form 0.01 to 0.99
    It also find the point on the curve that correspond to the current earth temperature and plot it.
    """

    #Calculating the surface temperature when varying the emissivity
    epsi_array=np.arange(0.01,1,0.01)
    t_earth_array=np.array([solve_N_layer(1,epsilon,s0=1350)[0] for epsilon in epsi_array])

    #print(t_earth_array[0], t_earth_array[-1])

    #Let's find the emissivity of the single-layer atmosphere to have a surface temperature of t_earth_surface=288K
    idx = np.argmin(np.abs(t_earth_array - 288))
    earth_epsilon = epsi_array[idx]
    print(f"To match Earth's surface temperature with a single layer, the emissivity should be {earth_epsilon:.2f}")

    #plotting the figure for Q3)a)
    fig,ax=plt.subplots(1,1)

    ax.plot(epsi_array,t_earth_array, label="Temperature of Earth's surface")

    ax.axhline(y=t_earth_surface, color='r', linestyle='--')
    ax.text(plt.xlim()[0], t_earth_surface, f" y = {t_earth_surface:.0f} K", ha="left", va="bottom", color="r", fontsize=10)

    ax.axvline(x=earth_epsilon, color='r', linestyle='--')
    ax.text(earth_epsilon, plt.ylim()[0], f" x = {earth_epsilon:.2f}", ha="left", va="bottom", color="r", fontsize=10)

    ax.set_xlabel("Emissivity of the atmosphere")
    ax.set_ylabel("Temperature of the atmosphere (K)")
    ax.set_title("Evolution of Earth's surface temperature wrt the emissivity of a single layer atmosphere")
    ax.legend()
    return None

def question3b():
    """
    This function answers the second part of question 3) by plotting the Earth's surface temperature in an 
    atmosphere with a number of layers from 1 to 10.
    It also find the point on the curve that correspond to the current earth temperature and plot it.
    Then it plots the temperature with ratio to the altitude in this particular atmosphere.
    """

    epsilon_earth=0.255

    #Calculating the surface temperature when varying the number of layers
    nb_layer_array=np.arange(1,11,1)
    t_earth_array2=np.array([solve_N_layer(nb_layer,epsilon_earth,s0=1350)[0] for nb_layer in nb_layer_array])

    #print(t_earth_array2[0], t_earth_array2[-1])

    #Let's find the number of layers for an atmosphere whose emissivity is 0.255 to have a surface temperature of t_earth_surface=288K
    idx = np.argmin(np.abs(t_earth_array2 - 288))
    earth_nb_layers = nb_layer_array[idx]
    print(f"To match Earth's surface temperature with an emissivity of {epsilon_earth}, earth should have {earth_nb_layers} layers")

    #Let's calculate the temperature of earth's atmosphere wrt the altitude 
    t_earth_layers=solve_N_layer(earth_nb_layers,epsilon_earth,s0=1350)
    altitudes=np.linspace(0,100,earth_nb_layers+1)

    #plotting the figures
    fig2,(ax21,ax22)=plt.subplots(2,1)
    ax21.plot(nb_layer_array,t_earth_array2)
    ax21.set_title("Temperature of Earth's surface wrt the number of layers")
    ax21.set_xlabel("Number of layers")
    ax21.set_ylabel("Temperature (K)")

    ax21.axhline(y=t_earth_surface, color='r', linestyle='--')
    ax21.text(plt.xlim()[0], t_earth_surface, f"{t_earth_surface:.0f} K", ha="left", va="bottom", color="r", fontsize=10)

    ax21.axvline(x=earth_nb_layers, color='r', linestyle='--')
    ax21.text(earth_nb_layers, plt.ylim()[0], f" x = {earth_nb_layers:.2f}", ha="left", va="bottom", color="r", fontsize=10)

    ax22.plot(t_earth_layers,altitudes)
    ax22.set_title(f"Temperatures in a {earth_nb_layers} layers atmosphere with an emissivity of {epsilon_earth} ")
    ax22.set_xlabel("Temperature (K)")
    ax22.set_ylabel("Altitude (km)")
    return None

def question4():
    """
    This function answers question 4) by plotting the Venuus' surface temperature in an 
    atmosphere with a number of layers from 1 to 20.
    It also find the point on the curve that correspond to the current Venus' surface temperature and plot it.
    """

    epsilon_venus=1 # each layer absorbs all the long wave energy
    t_venus=700 # K
    s0_venus=2600 # W/m2
    albedo_venus=0.71

    #Generating the surface temperatures for each nb of layer
    nb_layer_array=np.arange(1,100,1)
    t_venus_array=np.array([solve_N_layer(nb_layer,epsilon_venus,s0=s0_venus, alpha=albedo_venus)[0] for nb_layer in nb_layer_array])

    #print(t_venus_array[0],t_venus_array[-1])

    #Let's find the number of layers for an atmosphere whose emissivity is 0.71 to have a surface temperature of t_venus_surface=700K
    idx = np.argmin(np.abs(t_venus_array - t_venus))
    venus_nb_layers = nb_layer_array[idx]
    print(f"To match Venus' surface temperature with an emissivity of {epsilon_venus}, Venus should have {venus_nb_layers} layers")


    #Plotting the Temperature od Venus' surface wrt the number of layers 
    fig2,ax=plt.subplots(1,1)
    ax.plot(nb_layer_array,t_venus_array)
    ax.set_title("Temperature of Venus' surface wrt the number of layers")
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Temperature (K)")

    #Plotting the optimal number of layer to match venus' surface temperature

    ax.axhline(y=t_venus, color='r', linestyle='--')
    ax.text(plt.xlim()[0], t_venus, f" y = {t_venus:.0f} K", ha="left", va="bottom", color="r", fontsize=10)

    ax.axvline(x=venus_nb_layers, color='r', linestyle='--')
    ax.text(venus_nb_layers, plt.ylim()[0], f" x = {venus_nb_layers:.2f}", ha="left", va="bottom", color="r", fontsize=10)
    return None 

def question5():
    """
    This function answers  question 5) by plotting the Earth's surface temperature within the atmosphere 
    in case of a nuclear winter (all the sun radiation is absorbed by the last layer of atmosphere).
    """
    nb_layers=5
    epsilon_nuclear_winter=0.7
    alpha_layer=0.4

    altitudes=np.linspace(0,100,nb_layers+1)
    t_earth_layers=solve_N_layer_nuclear_winter(nb_layers,epsilon_nuclear_winter, alpha_layer)

    print(f"The Earth's surface temperature would be {t_earth_layers[0]:.2f} K during a nuclear winter")
    #print(t_earth_layers)

    #Plotting the Temperature within the Earth's atmosphere for the realistic number of layers
    fig,ax=plt.subplots(1,1)
    ax.plot(t_earth_layers,altitudes)
    ax.set_title(f"Temperatures in a 5 layers atmosphere during a nuclear winter")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Altitude (km)")
    T_earth=t_earth_layers[0]
    ax.text(T_earth,0,f'{T_earth:.0f}',color='red',va='top',ha='center')
    return None