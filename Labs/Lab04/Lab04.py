#! /usr/bin/env python3
'''
Lab 04 - Forest Fire and Disease Spread Models
This code implements a simple model to simulate the spread of forest fires and diseases.

Author: Adrien Boissy
Collaborators: None


TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
•   ipythonexit()
•   run Lab04.py
•   plt.ion()
•   methodology()
•   question_1() : to validate the solver with two simple examples, in Q1
•   question_2() : to answer Q2
•   question_3() : to answer Q3



'''
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d
plt.style.use('seaborn-v0_8-darkgrid')

#Select colors at : https://www.w3schools.com/colors/colors_picker.asp 

#define the colormap for the forest fire model
colors_forest= ['tan', 'forestgreen','crimson']
forest_cmap = ListedColormap(colors_forest)

#define the colormap for the disease spread model
colors_disease= ['black', 'deepskyblue','mediumseagreen','orangered']
disease_cmap = ListedColormap(colors_disease)



def forest_disease_solver(isize=3,jsize=3,nstep=4,p_spread=1.,p_ignite=None,
                            p_bare=None,p_fatal=0.,debug=False):
    '''
    Create a forest fire or disease spread model, where fire/disease can spread from neighbor to neighbor with a given probability.
    The model can simulate both a forest fire and a disease spread by adjusting the parameters.

    Forest fire states:
    1: Bare/burned
    2: Forested
    3: Burning
    Disease spread states:
    0: Dead
    1: Immune
    2: Healthy
    3: Infected
    ----------------
    Parameters:
    isize, jsize, nstep : int
        Size of the grid and number of time steps
    p_spread : float
        Probability of fire/disease spread from neighbor to neighbor
    p_ignite : float or None
        Initial probability of a cell being on fire/infected at time 0
        If None, the initial fire/infected person is set to the center of the grid
    p_bare : float or None
        Initial probability of a cell being bare/immune at time 0
        If None, all cells are forested/healthy at time 0
    p_fatal : float
        Probability of an infected person to die
        set to 0 to simulate a forest fire
    debug : bool
        If True, print if the number of dead cells. 
        Used to verify that no cell is marked as dead in the forest fire model
    
    ---------------
    Returns:
    grid : 3d array
        3D array of shape (nstep, isize, jsize) representing the state of the grid over time
    ---------------
    '''
    #Creating a forest/persons grid and making all spots have trees
    grid=np.zeros((nstep,isize,jsize),dtype=int) +2

    if p_bare is not None:
        bare_spots=random.rand(isize,jsize)<=p_bare
        grid[0,bare_spots]=1 

    if p_ignite is not None:
        fire_infected_cells=random.rand(isize,jsize)<=p_ignite
        grid[0,fire_infected_cells]=3
    else:
        #Set the initial fire/infected person to the center
        grid[0,isize//2,jsize//2]=3
    

    for k in range(0,nstep-1):
        #Assume at the next time step the grid is the same as current time step
        grid[k+1,:,:]=grid[k,:,:]
        propagation_front=grid[k,:,:]==3
        sane_cells=grid[k,:,:]==2 #The cells that are healthy/forested

        #counting for each cell the number of neighbors burning/infected using convolution
        kernel = np.array([[0,1,0],
                           [1,0,1],
                           [0,1,0]])

        neighbors_on_fire_infected= convolve2d(propagation_front, kernel, mode='same', boundary='fill', fillvalue=0)

        #computing the probability of a cell to catch fire/infection based on the number of its neighbors on fire/infected
        probs_to_propagate= 1 - (1-p_spread)**neighbors_on_fire_infected
        #rolling a dice for all cells
        probs=random.rand(isize,jsize)

        grid[k+1,sane_cells & (probs <= probs_to_propagate)]=3  #Set tree/healthy cells on fire/infected based on their respective probability to catch fire/infection
        fatal_probability=random.rand(isize,jsize)
        grid[k+1,propagation_front]=1  #Burn the current fire spots. First consider that every person infected recovers
        grid[k+1,propagation_front & (fatal_probability < p_fatal)]=0  #Some of the people that were infected will in fact die
        
        if debug:
            #Let's verify that when using the solver to solve forest fire, no cell is marked as 0 (dead)
            n_dead_cells= np.sum(grid[k+1,:,:]==0)
            if n_dead_cells>0:
                print(f"Debug info: At time step {k+1}, there are {n_dead_cells} dead cells in the forest.")
    return grid


def plot_forest2d(forest_in,itime=0):
    '''
    This function makes a 2D plot of the forest at a given time step,
    This function plots the forest grid in an optimal way when the grid is squared.
    If the grid is not squared, the function should be modified.

    -----------
    Parameters:
    forest_in : 3d array
        3D array of shape (ntime, isize, jsize) representing the state of the forest over time
    itime : int
        Time step to plot

    -----------
    Returns:
    fig, ax : matplotlib figure and axis
        Figure and axis objects of the plot
    '''
    fig, ax = plt.subplots(1,1, figsize=(6,8))

    my_map = ax.pcolor(forest_in[itime,:,:],vmin=1,vmax=3,cmap=forest_cmap)

    cbar=plt.colorbar(my_map,ax=ax, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax.invert_yaxis()
    #Add labels and title
    ax.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax.set_title(f'The Seven Acre wood at T={itime:03d}')

    return fig,ax


def plot_disease2d(grid_in,itime=0):
    '''
    This function makes a 2D plot of the disease spread at a given time step,
    This function plots the disease grid in an optimal way when the grid is squared.
    If the grid is not squared, the function should be modified.

    -----------
    Parameters:
    grid_in : 3d array
        3D array of shape (ntime, isize, jsize) representing the state of the disease spread over time
    itime : int
        Time step to plot
        
    -----------
    Returns:
    fig, ax : matplotlib figure and axis
        Figure and axis objects of the plot
    '''
    fig, ax = plt.subplots(1,1, figsize=(6,8))

    my_map = ax.pcolor(grid_in[itime,:,:],vmin=0,vmax=3,cmap=disease_cmap)

    cbar=plt.colorbar(my_map,ax=ax, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([0,1,2,3])
    cbar.set_ticklabels(['Dead','Immune','Healthy','Infected'])

    #Invert y axis to match matrix orientation
    ax.invert_yaxis()
    #Add labels and title
    ax.set_xlabel('Eastward(persons) $\\longrightarrow$')
    ax.set_ylabel('Northward (persons) $\\longrightarrow$')
    ax.set_title(f'The covid Party at t={itime:03d}')

    return fig,ax


def results(initial_value=0,final_value=1, nb_values=11, variable='p_spread',solver=forest_disease_solver,**kwargs):
    '''
    This function runs the chosen solver for a range of value of a given variable and collects the initial and final states.
    
    -----------
    Parameters:
    initial_value : float, default 0
        Initial value of the variable to be varied

    final_value : float, default 1
        Final value of the variable to be varied

    nb_values : int, default 11
        Number of values to be tested between initial and final value

    variable : str, default 'p_spread'
        Name of the variable to be varied

    solver : function, default forest_disease_solver
        Solver function to be used 

    kwargs : dict
        Additional parameters to be passed to the solver function
    -----------
    Returns:
    param_values : 1d array
        Array of the values of the variable

    initial_fire_infected : 1d array
        Initial percentage of fire/infected for each parameter value

    initial_sane : 1d array
        Initial percentage of sane/healthy for each parameter value

    initial_dead : 1d array
        Initial percentage of dead for each parameter value

    final_bare_immune : 1d array
        Final percentage of bare/immune for each parameter value

    final_sane : 1d array
        Final percentage of sane/healthy for each parameter value

    final_dead : 1d array
        Final percentage of dead for each parameter value
    '''
    npoints=kwargs['isize']*kwargs['jsize']

    #Creating arrays to store the results for each value of the variable
    param_values=np.linspace(initial_value,final_value,nb_values)
    initial_fire_infected=np.zeros(nb_values)
    initial_sane=np.zeros(nb_values) 
    initial_dead=np.zeros(nb_values)
    final_bare_immune=np.zeros(nb_values)
    final_sane=np.zeros(nb_values)
    final_dead=np.zeros(nb_values)


    for index,param in enumerate(param_values):
        kwargs[variable]=param
        result=solver(**kwargs)

        #Calculating initial percentage of fire/infected, sane/healthy
        loc = result[0,:,:] == 3
        initial_fire_infected[index] = 100 * loc.sum()/npoints

        loc = result[0,:,:] == 2
        initial_sane[index] = 100 * loc.sum()/npoints
        
        ##Calculating final percentage of bare/immune, sane/healthy, dead
        loc = result[-1,:,:] == 1
        final_bare_immune[index] = 100 * loc.sum()/npoints

        loc = result[-1,:,:] == 2
        final_sane[index] = 100 * loc.sum()/npoints

        loc = result[-1,:,:] == 0
        final_dead[index] = 100 * loc.sum()/npoints

    return param_values, initial_fire_infected, initial_sane, initial_dead, final_bare_immune, final_sane, final_dead
        

def make_all_2dplots(forest_in,folder='Labs/Lab04/results/',plot_function=plot_forest2d):
    '''
    Make all 2D plots for all time steps, and save them in the specified folder. 
    The plot_function parameter allows to choose between forest and disease plots.
    -----------
    Parameters:
    forest_in : 3d array
        3D array of shape (ntime, isize, jsize) representing the state of the forest/disease over time
    folder : str, default 'Labs/Lab04/results/'
        Folder where the plots will be saved
    plot_function : function, default plot_forest2d
        Function used to create the plots
    
    '''
    import os

    #Check if folder exists, if not make it
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    #make a bunch of plots
    ntime,nx,ny=forest_in.shape
    for i in range(ntime):
        fig,ax = plot_function(forest_in,itime=i)
        fig.savefig(f"{folder}/forest_i{i:04d}.png")
        plt.close('all')

    
def plot_progression(forest,additional_title=''):
    '''
    Calculate the time dynamics of a forest fire and plot them.
    -----------
    Parameters:
    forest : 3d array
        3D array of shape (ntime, isize, jsize) representing the state of the forest/disease over time
    additional_title : str, default ''
        Additional title to be added to the plot
    '''

    # Get total number of points:
    ksize, isize, jsize = forest.shape
    npoints = isize * jsize

    # Find all spots that have forests (or are healthy people)
    # ...and count them as a function of time.
    loc = forest == 2
    forested = 100 * loc.sum(axis=(1, 2))/npoints

    loc = forest == 1
    bare = 100 * loc.sum(axis=(1, 2))/npoints

    fig,ax=plt.subplots(1,1, figsize=(8,6))
    ax.plot(forested, label='Forested')
    ax.plot(bare, label='Bare/Burnt')
    ax.set_xlabel('Time (arbitrary units)')
    ax.set_ylabel('Percent Total Forest')
    ax.set_title(f'Forest Fire Progression Over Time' + additional_title)
    ax.legend()
    plt.show()
    return fig,ax


def test_neighbors(nx,ny):
    forest=np.zeros((1,nx,ny))
    forest[0,random.rand(nx,ny)>0.5]=3
    print(forest)

    north_on_fire= np.roll(forest[0],1,axis=0)==3
    north_on_fire[0,:]=False  #First row has no north neighbor

    south_on_fire= np.roll(forest[0],-1,axis=0)==3
    south_on_fire[-1,:]=False  #First row has no south neighbor
    
    east_on_fire= np.roll(forest[0],-1,axis=1)==3
    east_on_fire[:,-1]=False  #First row has no east neighbor
    
    west_on_fire= np.roll(forest[0],1,axis=1)==3
    west_on_fire[:,0]=False  #First row has no west neighbor

    neighbors_on_fire= north_on_fire.astype(int) + south_on_fire.astype(int) + east_on_fire.astype(int) + west_on_fire.astype(int)
    print(neighbors_on_fire)

    fire_front=forest[0,:,:]==3

    kernel = np.array([[0,1,0],
                           [1,0,1],
                           [0,1,0]])

    neighbors_on_fire_convolve= convolve2d(fire_front, kernel, mode='same', boundary='fill', fillvalue=0)
    print(neighbors_on_fire_convolve)
    print(np.array_equal(neighbors_on_fire, neighbors_on_fire_convolve))


def methodology():
    '''
    This function plot the figure used in the methodology section of the report.
    The first and second time steps of a 3x3 forest with an initial fire in the center

    '''
    #First example: 3x3 forest with initial fire in center
    forest1=forest_disease_solver(isize=3,jsize=3,nstep=4,p_spread=1.0)
    fig,(ax1,ax2)=plt.subplots(1,2, figsize=(10,6))
    fig.suptitle('3x3 Forest: central ignition and sure propagation', fontsize=16)
    
    #Forest at time 0
    my_map0 = ax1.pcolor(forest1[0,:,:],vmin=1,vmax=3,cmap=forest_cmap)

    cbar=plt.colorbar(my_map0,ax=ax1, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax1.invert_yaxis()
    #Add labels and title
    ax1.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax1.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax1.set_title(f'The Seven Acre wood at T=000')
    #Forest at time 1
    my_map1 = ax2.pcolor(forest1[1,:,:],vmin=1,vmax=3,cmap=forest_cmap)

    cbar=plt.colorbar(my_map1,ax=ax2, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax2.invert_yaxis()
    #Add labels and title
    ax2.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax2.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax2.set_title(f'The Seven Acre wood at T=001')

    plt.show()


def question_1():
    '''
    Answer question 1 of Lab4. Validation of the forest fire model in two examples:
    - 3x3 frid with an initial fire in the center
    - 8x3 grid with an initial fire in the center
    '''
    #First example: 3x3 forest with initial fire in center
    forest1=forest_disease_solver(isize=3,jsize=3,nstep=4,p_spread=1.0)
    fig,((ax11,ax12),(ax21,ax22))=plt.subplots(2,2, figsize=(7,8))
    fig.suptitle('3x3 Forest: central ignition and sure propagation', fontsize=16)
    
    #Forest at time 0
    my_map0 = ax11.pcolor(forest1[0,:,:],vmin=1,vmax=3,cmap=forest_cmap)

    cbar=plt.colorbar(my_map0,ax=ax11, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax11.invert_yaxis()
    #Add labels and title
    ax11.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax11.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax11.set_title(f'The Seven Acre wood at T=000')

    #Forest at time 1
    my_map1 = ax12.pcolor(forest1[1,:,:],vmin=1,vmax=3,cmap=forest_cmap)

    cbar=plt.colorbar(my_map1,ax=ax12, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax12.invert_yaxis()
    #Add labels and title
    ax12.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax12.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax12.set_title(f'The Seven Acre wood at T=001')

    #Forest at time 2
    my_map2 = ax21.pcolor(forest1[2,:,:],vmin=1,vmax=3,cmap=forest_cmap)

    cbar=plt.colorbar(my_map2,ax=ax21, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax21.invert_yaxis()
    #Add labels and title
    ax21.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax21.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax21.set_title(f'The Seven Acre wood at T=002')

    #Forest at time 3
    my_map3 = ax22.pcolor(forest1[3,:,:],vmin=1,vmax=3,cmap=forest_cmap)

    cbar=plt.colorbar(my_map3,ax=ax22, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax22.invert_yaxis()
    #Add labels and title
    ax22.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax22.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax22.set_title(f'The Seven Acre wood at T=003')


    #Second example: 8x3 forest with initial fire in center
    forest2=forest_disease_solver(isize=3,jsize=6,nstep=6,p_spread=1.0)

    fig2,((ax2_11,ax2_12,ax2_13),(ax2_21,ax2_22,ax2_23))=plt.subplots(2,3, figsize=(14,7))
    fig2.suptitle('3x6 Forest: central ignition and sure propagation', fontsize=16)

    #Forest at time 0
    my_map0 = ax2_11.pcolor(forest2[0,:,:],vmin=1,vmax=3,cmap=forest_cmap)
    cbar=plt.colorbar(my_map0,ax=ax2_11, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax2_11.invert_yaxis()
    #Add labels and title
    ax2_11.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax2_11.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax2_11.set_title(f'The Seven Acre wood at T=000')
    #Forest at time 1
    my_map1 = ax2_12.pcolor(forest2[1,:,:],vmin=1,vmax=3,cmap=forest_cmap)

    cbar=plt.colorbar(my_map1,ax=ax2_12, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax2_12.invert_yaxis()
    #Add labels and title
    ax2_12.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax2_12.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax2_12.set_title(f'The Seven Acre wood at T=001')
    #Forest at time 2
    my_map2 = ax2_13.pcolor(forest2[2,:,:],vmin=1,vmax=3,cmap=forest_cmap)

    cbar=plt.colorbar(my_map2,ax=ax2_13, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax2_13.invert_yaxis()
    #Add labels and title
    ax2_13.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax2_13.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax2_13.set_title(f'The Seven Acre wood at T=002')
    #Forest at time 3
    my_map3 = ax2_21.pcolor(forest2[3,:,:],vmin=1,vmax=3,cmap=forest_cmap)

    cbar=plt.colorbar(my_map3,ax=ax2_21, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax2_21.invert_yaxis()
    #Add labels and title
    ax2_21.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax2_21.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax2_21.set_title(f'The Seven Acre wood at T=003')
    #Forest at time 4
    my_map4 = ax2_22.pcolor(forest2[4,:,:],vmin=1,vmax=3,cmap=forest_cmap)        

    cbar=plt.colorbar(my_map4,ax=ax2_22, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax2_22.invert_yaxis()
    #Add labels and title
    ax2_22.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax2_22.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax2_22.set_title(f'The Seven Acre wood at T=004')
    #Forest at time 5
    my_map5 = ax2_23.pcolor(forest2[5,:,:],vmin=1,vmax=3,cmap=forest_cmap)        

    cbar=plt.colorbar(my_map5,ax=ax2_23, shrink=.8, fraction=.08, location='bottom', orientation='horizontal')
    cbar.set_ticks([1,2,3])
    cbar.set_ticklabels(['Bare/burned','Forested','Burning'])

    #Invert y axis to match matrix orientation
    ax2_23.invert_yaxis()
    #Add labels and title
    ax2_23.set_xlabel('Eastward($km$) $\\longrightarrow$')
    ax2_23.set_ylabel('Northward ($km$) $\\longrightarrow$')
    ax2_23.set_title(f'The Seven Acre wood at T=005')
    


    plt.show()


def question_2():
    '''
    This function analyzes the effect of varying the spread probability of spreading 
    and the initial percentage of bare cells on the burning of the forest.
    The results are plotted in two subplots, and used in the report to answer question 2.
    It may not reproduce exactly the same values as in the report due to the stochastic nature of the model.
    '''

    kwargs=dict(isize=100,jsize=100,nstep=300,p_spread=1,p_ignite=0.02)

    #Varying the spread probability of fire from 0 to 1
    p_spread_values, initial_fires, initial_forests, initial_dead, final_bare, final_forest, final_dead = results(**kwargs)
    
    fig,ax=plt.subplots(1,2, figsize=(12,6))
    ax[0].plot(p_spread_values, final_bare,label='Final Bare/burned',color='tan')
    ax[0].plot(p_spread_values, final_forest,label='Final Forest',color='forestgreen')
    ax[0].plot(p_spread_values, initial_forests, label='Initial Forest', color='darkgreen', linestyle='--')
    ax[0].plot(p_spread_values, initial_fires, label='Initial Fires', color='crimson', linestyle='--')
    ax[0].set_xlabel('Spread Probability')
    ax[0].set_ylabel(f'Percentage of the grid (%)')
    ax[0].set_title(f'Burning of the forest depends on the spread probability')
    ax[0].legend()

    #Varying the initial percentage of bare cells from 0 to 100%
    p_ignite_values, initial_fires, initial_forests, initial_dead, final_bare, final_forest, final_dead = results(variable='p_bare',**kwargs)

    ax[1].plot(p_ignite_values*100, final_bare,label='Final Bare/burned',color='tan')
    ax[1].plot(p_ignite_values*100, final_forest,label='Final Forest',color='forestgreen')
    ax[1].plot(p_ignite_values*100, initial_forests, label='Initial Forest', color='darkgreen', linestyle='--')
    ax[1].plot(p_ignite_values*100, initial_fires, label='Initial Fires', color='crimson', linestyle='--')   
    ax[1].set_xlabel('Initial percentage of bare cells (%)')
    ax[1].set_ylabel(f'Percentage of the grid (%)')
    ax[1].set_title(f'Burning of the forest depends on initial percentage of bare cells')
    ax[1].legend()

    plt.show()


def question_3():
    '''
    This function analyzes the effect of varying the fatality probability of the disease
    and the initial percentage of immune people on the spread of the disease.
    The results are plotted in two subplots, and used in the report to answer question 3.
    It may not reproduce exactly the same values as in the report due to the stochastic nature of the model.
    '''
    kwargs=dict(isize=100,jsize=100,nstep=300,p_spread=0.5,p_ignite=0.02,p_fatal=1)
    p_fatal_values, initial_infected, initial_healthy, initial_dead, final_immune, final_healthy, final_dead = results(variable='p_fatal', **kwargs)

    fig,ax=plt.subplots(1,2, figsize=(12,6))
    
    ax[0].plot(p_fatal_values, initial_infected, label='Initial infected', color='orangered', linestyle='--')
    ax[0].plot(p_fatal_values, initial_healthy-final_healthy + initial_infected,label='Overall Infected people',color='orangered')
    ax[0].plot(p_fatal_values, final_immune,label='Final Immune',color='deepskyblue')
    ax[0].plot(p_fatal_values, final_dead,label='Final Dead',color='black')
    ax[0].set_xlabel('Fatality Probability')
    ax[0].set_ylabel(f'Percentage of the people (%)')
    ax[0].set_title(f"The probability of death doesn't affect the disease spread \n but it increases the death toll")
    ax[0].legend()
    
    p_fatal_values, initial_infected, initial_healthy, initial_dead, final_immune, final_healthy, final_dead = results(variable='p_bare', **kwargs)
    ax[1].plot(p_fatal_values, initial_healthy-final_healthy + initial_infected,label='Overall Infected people',color='orangered',linewidth=4)
    ax[1].plot(p_fatal_values, final_dead,label='Death toll',color='black')
    ax[1].plot(p_fatal_values, final_healthy+final_immune,label='Final Healthy + Immune',color='deepskyblue')
    ax[1].plot(p_fatal_values, initial_infected, label='Initial infected', color='orangered', linestyle='--')
    ax[1].plot(p_fatal_values, final_healthy,label='Final Healthy',color='mediumseagreen')
    ax[1].set_xlabel('Percentage of vaccinated people (%)')
    ax[1].set_ylabel(f'Percentage of people (%)')
    ax[1].set_title(f'The more people are vaccinated, the less the disease spreads\n p_fatal=1')
    ax[1].legend()
    
    plt.show()


def animate_forest_fire(folder='Labs/Lab04/results/', ntime=10, interval=500):
    '''
    This function creates an animation of the forest fire from the saved images in the specified folder.
    This function is not used in the report, but can be used to visualize the forest fire simulation.
    -----------
    Parameters:
    folder : str
        Folder where the images are saved
    ntime : int
        Number of time steps/images to animate
    interval : int
        Interval between frames in milliseconds
    -----------
    Returns:
    ani : matplotlib animation
        Animation object
    '''
    import os
    import matplotlib.animation as animation
    from matplotlib.image import imread

    fig, ax = plt.subplots(1,1, figsize=(6,6))

    def update(frame):
        ax.clear()
        img = imread(f"{folder}/forest_i{frame:04d}.png")
        ax.imshow(img)
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=ntime, interval=interval)
    plt.show()
    return ani


def forest_fire(isize=3,jsize=3,nstep=4,p_spread=1.):
    '''
    Brute force implementation of a forest fire model, where fire can spread from neighbor to neighbor with a given probability.
    ---------------
    Parameters:
    isize, jsize, nstep : int
        Size of the grid and number of time steps
    p_spread : float
        Probability of fire spread from neighbor to neighbor
    ---------------
    Returns:
    forest : 3d array 
    '''
    #Creating a forest and making all spots have trees
    forest=np.zeros((nstep,isize,jsize),dtype=int) +2

    #Set initial fire to center [NEED TO BE CHANGED LATER]
    forest[0,isize//2,jsize//2]=3

    for k in range(0,nstep-1):
        #Assume the next time step is the same as current
        forest[k+1,:,:]=forest[k,:,:]
        for i in range(isize):
            for j in range(jsize):
                if forest[k,i,j]==3:
                    rd_north=random.rand()
                    rd_south=random.rand()
                    rd_west=random.rand()
                    rd_east=random.rand()

                    #Should we spread fire on North
                    if ((rd_north <=p_spread) and (i>0) and (forest[k,i-1,j]==2)):
                        forest[k+1,i-1,j]=3

                    #Should we spread fire on South
                    if ( rd_south <=p_spread and (i<isize-1) and (forest[k,i+1,j]==2)):
                        forest[k+1,i+1,j]=3

                    #Should we spread fire on West
                    if ( rd_west <=p_spread and (j>0) and (forest[k,i,j-1]==2)):
                        forest[k+1,i,j-1]=3

                    #Should we spread fire on East
                    if ( rd_east <=p_spread and (j<jsize-1) and (forest[k,i,j+1]==2)):
                        forest[k+1,i,j+1]=3
                    
                    #The burning cell becomes bare/burned
                    forest[k+1,i,j]=1 

    return forest
