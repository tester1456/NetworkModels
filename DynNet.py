#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:08:02 2017

@authors: Rohit Konda, Vineet Tiruvadi
#Modules for working with dynamical network models
Note:
    Might be a smart idea to merge with TVB modules if they already do a lot of this
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def unity(x):
    return x     

class NetworkModel:
    coupling = []
    
    def __init__(self, A, K = 10, dt = .01):
        self.G = nx.from_numpy_matrix(A) #Graph
        self.states = np.matrix([4,1,3,5,6,2]).T #np.matrix(np.random.uniform(0,2*np.pi,size=(6,1)))# memory of phases
        self.w = np.matrix([3.0,3.3,3.6,3.9,4.2,4.5]).T#np.matrix(np.random.normal(3,.2,size=(6,1))) #init intrinsic freq.
        self.t = 0 #time
        self.K = K #coupling constant
        self.dt = dt #time step
        self.coupling = unity
    
    def set_coupling(self,coupling_class):
        if coupling_class == 'sin':
            self.coupling = np.sin
        elif coupling_class == 'tanh':
            self.coupling = np.tanh
    
    #Kuramoto differential equation
    def phase_dev(self,phase):
        D = (nx.incidence_matrix(self.G, oriented = True, weight = 'weight')).todense() #incidence
        N = np.random.normal(0, 10, [len(D[0]), 1])
        return self.w - self.K / len(self.G) * D * self.coupling(D.T * self.states[:,-1]) # + N
    
    # 4th order Runge-Kutta approximation
    def step(self):
        new_state = self.states[:,-1] + self.phase_dev(self.states[:,-1])*self.dt
        new_state = new_state % (2 * np.pi)
        self.t += self.dt
        self.states = np.hstack((self.states,new_state))

def run_model(A, K = 10, t = 10):
    P = NetworkModel(A, K)
    for ts in range(0,int(t/P.dt)):
        P.step()
    return P

def state_course(states):
    plt.figure()
    plt.plot(np.sin(states.T))
    plt.xlabel('Time Steps')
    plt.ylabel('Sin of Phase')
    plt.title('State of Nodes')
    plt.show()

#%% Run script

if __name__ == '__main__':
    nodes = [1,2,3,4,5,6]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from([(1,2,1),(2,3,1),(1,6,1),(1,3,1),(3,4,1),(4,5,1),(4,6,1),(5,6,1)])
    
    
    #Quick plot of G
    nx.draw(G)
    A = nx.to_numpy_matrix(G)
    state_course(run_model(A).states)