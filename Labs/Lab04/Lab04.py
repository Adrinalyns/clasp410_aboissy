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

nstep=10
ny=10
nx=10
forest=np.zeros((nstep,ny,nx),dtype=int) +2

propagation_front=[]
while len(propagation_front)>0:
    
    for grid_on_fire in propagation_front:
        forest=spreading_fire(forest,grid_on_fire)
        forest[grid_on_fire]=1
        #remove the cell from propagation front

def spreading_fire(forest,grid_on_fire,prob_fire):
    '''
    Try to spread the fire on the 4 adjacent cells if they are inside the grid
    '''
    k,i,j=grid_on_fire
    nstep,isize,jsize=forest.shape

    rd_north=random.rand()
    rd_south=random.rand()
    rd_west=random.rand()
    rd_east=random.rand()

    #Should we spread fire on North
    if ((rd_north <=prob_fire) and (i>0) and (forest[k,i-1,j]==2)):
        forest[k+1,i-1,j]=3

    #Should we spread fire on South
    if ( rd_south <=prob_fire and (i<isize-1) and (forest[k,i+1,j]==2)):
        forest[k+1,i+1,j]=3

    #Should we spread fire on West
    if ( rd_west <=prob_fire and (j>0) and (forest[k,i,j-1]==2)):
        forest[k+1,i,j-1]=3

    #Should we spread fire on East
    if ( rd_east <=prob_fire and (j<jsize-1) and (forest[k,i,j+1]==2)):
        forest[k+1,i,j+1]=3

    return forest
    




def forest_fire(isize=3,jsize=3,nstep=4):
    '''
    create a forest fire
    '''
    #Creating a forest and making all spots have trees
    forest=np.zeros((nstep,isize,jsize),dtype=int) +2

    #Set initial fire to center [NEED TO BE CHANGED LATER]
    forest[0,isize//2,jsize//2]=3
    print(forest)

    for k in range(0,nstep-1):
        #Assume the next time step is the same as current
        forest[k+1,:,:]=forest[k,:,:]
        for i in range(isize):
            for j in range(jsize):
                if forest[k,i,j]==3:
                    forest=spreading_fire(forest,(k,i,j),1.)
                    forest[k+1,i,j]=1 

    return forest
