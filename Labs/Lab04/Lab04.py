#! /usr/bin/env python3
'''
Author: Adrien Boissy
Collaborators: None


TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
•	run lab03.py
•	plt.ion()
•	question_1() : to validate the solver and obtain the plot, in Q1
•	question_2() : to obtain all the plots used for answering Q2
•	question_3() : to obtain all the plots used for answering Q3


'''
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d
plt.style.use('seaborn-v0_8-darkgrid')

#Select colors at : https://www.w3schools.com/colors/colors_picker.asp 

colors_forest= ['tan', 'forestgreen','crimson']
forest_cmap = ListedColormap(colors_forest)

colors_disease= ['black', 'deepskyblue','mediumseagreen','orangered']
disease_cmap = ListedColormap(colors_disease)



def forest_fire(isize=3,jsize=3,nstep=4,p_spread=1.):
    '''
    create a forest fire
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



def forest_disease_solver(isize=3,jsize=3,nstep=4,p_spread=1.,p_ignite=None,
                            p_bare=None,p_fatal=0.,debug=False):
    '''
    create a forest fire
    '''
    #Creating a forest and making all spots have trees
    forest=np.zeros((nstep,isize,jsize),dtype=int) +2
    if p_bare is not None:
        bare_spots=random.rand(isize,jsize)<=p_bare
        forest[0,bare_spots]=1 

    if p_ignite is not None:
        fire_spots=random.rand(isize,jsize)<=p_ignite
        forest[0,fire_spots]=3
    else:
        #Set the initial fire only to center
        forest[0,isize//2,jsize//2]=3

    for k in range(0,nstep-1):
        #Assume the next time step is the same as current
        forest[k+1,:,:]=forest[k,:,:]
        fire_front=forest[k,:,:]==3
        trees=forest[k,:,:]==2

        #counting for each cell the number of neighbors on fire using convolution
        kernel = np.array([[0,1,0],
                           [1,0,1],
                           [0,1,0]])

        neighbors_on_fire= convolve2d(fire_front, kernel, mode='same', boundary='fill', fillvalue=0)

        #computing the probability of a cell to catch fire based on the number of its neighbors on fire
        probs_to_propagate= 1 - (1-p_spread)**neighbors_on_fire
        #rolling a dice for all cells
        probs=random.rand(isize,jsize)

        forest[k+1,trees & (probs <= probs_to_propagate)]=3  #Set trees on fire based on their respective probability to catch fire
        fatal_probability=random.rand(isize,jsize)
        forest[k+1,fire_front]=1  #Burn the current fire spots/everyone that was infected recovers
        forest[k+1,fire_front & (fatal_probability < p_fatal)]=0  #Some of the people that were infected finally die
        
        if debug:
            #Let's verify that when using the solver to solve forest fire, no cell is marked as 0 (dead)
            n_dead_cells= np.sum(forest[k+1,:,:]==0)
            if n_dead_cells>0:
                print(f"Debug info: At time step {k+1}, there are {n_dead_cells} dead cells in the forest.")
    return forest



def plot_forest2d(forest_in,itime=0):
    '''
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
    '''
    npoints=kwargs['isize']*kwargs['jsize']
    
    param_values=np.linspace(initial_value,final_value,nb_values)
    initial_fires=np.zeros(nb_values)
    initial_forests=np.zeros(nb_values)
    final_bare_wrt_param=np.zeros(nb_values)
    final_forest_wrt_param=np.zeros(nb_values)


    for index,param in enumerate(param_values):
        kwargs[variable]=param
        result=solver(**kwargs)

        loc = result[0,:,:] == 3
        initial_fires[index] = 100 * loc.sum()/npoints

        loc = result[0,:,:] == 2
        initial_forests[index] = 100 * loc.sum()/npoints
        
        loc = result[-1,:,:] == 1
        final_bare_wrt_param[index] = 100 * loc.sum()/npoints

        loc = result[-1,:,:] == 2
        final_forest_wrt_param[index] = 100 * loc.sum()/npoints

    return param_values, initial_fires, initial_forests, final_bare_wrt_param, final_forest_wrt_param
        

def make_all_2dplots(forest_in,folder='Labs/Lab04/results/',plot_function=plot_forest2d):
    '''
    Make all 2D plots for all time steps
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
    '''Calculate the time dynamics of a forest fire and plot them.'''

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

    fig2,((ax2_11,ax2_12,ax2_13),(ax2_21,ax2_22,ax2_23))=plt.subplots(2,3, figsize=(14,7.5))
    fig2.suptitle('8x6 Forest: central ignition and sure propagation', fontsize=16)

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

    #Varying p_spread from 0 to 1
    kwargs=dict(isize=20,jsize=20,nstep=30,p_spread=0.5,p_ignite=0.02)

    p_spread_values, initial_fires_wrt_p_spread, initial_forests_wrt_p_spread, final_bare_wrt_p_spread, final_forest_wrt_p_spread = results(**kwargs)
    p_ignite_values, initial_fires_wrt_p_ignite, initial_forests_wrt_p_ignite, final_bare_wrt_p_ignite, final_forest_wrt_p_ignite = results(variable='p_bare',**kwargs)

    fig,ax=plt.subplots(1,2, figsize=(12,6))
    ax[0].plot(p_spread_values, final_bare_wrt_p_spread,label='Final Bare/burned',color='tan')
    ax[0].plot(p_spread_values, final_forest_wrt_p_spread,label='Final Forest',color='forestgreen')
    ax[0].plot(p_spread_values, initial_forests_wrt_p_spread, label='Initial percentage of forested cells (%)', color='darkgreen', linestyle='--')
    ax[0].plot(p_spread_values, initial_fires_wrt_p_spread, label='Initial percentage of burning cells (%)', color='crimson', linestyle='--')
    ax[0].set_xlabel('Spread Probability')
    ax[0].set_ylabel(f'Percentage of the grid (%)')
    ax[0].set_title(f'Burning of the forest depends on the spread probability')
    ax[0].legend()

    ax[1].plot(p_ignite_values*100, final_bare_wrt_p_ignite,label='Final Bare/burned',color='tan')
    ax[1].plot(p_ignite_values*100, final_forest_wrt_p_ignite,label='Final Forest',color='forestgreen')
    ax[1].plot(p_ignite_values*100, initial_forests_wrt_p_ignite, label='Initial percentage of forested cells (%)', color='darkgreen', linestyle='--')
    ax[1].plot(p_ignite_values*100, initial_fires_wrt_p_ignite, label='Initial percentage of burning cells (%)', color='crimson', linestyle='--')   
    ax[1].set_xlabel('Initial percentage of bare cells (%)')
    ax[1].set_ylabel(f'Percentage of the grid (%)')
    ax[1].set_title(f'Burning of the forest depends on initial percentage of bare cells')
    ax[1].legend()

    plt.show()


    


def animate_forest_fire(folder='Labs/Lab04/results/', ntime=10, interval=500):
    '''
    Create an animation from the saved plots
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