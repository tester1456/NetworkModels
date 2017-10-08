#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:08:02 2017

@authors: Rohit Konda, Vineet Tiruvadi
# Creation of different brain network models - Kuramoto, WC , and NMM
Notes:
    Outline of 3 computational models that we are using
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import scipy.signal as sig
import scipy.stats as stats
import statsmodels
import network_modules as nm
import pickle


# generate sparse graph , n: number of nodes, lam: lambda for poisson
# distribution for number of edges, bias: 0
# for small world properties, and infinite for a random graph
def sparse_BA(n, lam, bias):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for node in G:
        n_edges = round(np.random.poisson(lam))
        if n_edges >= n:  # add edges to all other nodes
            i_vec = [node] * (n-1)
            j_vec = [x for x in range(n) if x != node]
            edges = [(i, j) for i, j in zip(i_vec, j_vec)]
            for i in range(len(edges)):
                if np.random.random() > .5:  # randomize direction
                    edges[i] = (edges[i][1], edges[i][0])
            G.add_edges_from(edges)
        else:
            i_vec = [node] * n_edges
            j_vec = []
            degrees = np.array(list(nx.degree(G).values())) + bias
            degrees[node] = 0
            degrees = (degrees)/sum(degrees)
            cs = np.cumsum(degrees)
            for i in range(n_edges):
                r = np.random.random()
                ind, = np.where(cs > r)  # add edges based on degree of nodes
                j_vec.append(ind[0])
            edges = [(i, j) for i, j in zip(i_vec, j_vec)]
            for i in range(len(edges)):
                if np.random.random() > .5:  # randomize direction
                    edges[i] = (edges[i][1], edges[i][0])
            G.add_edges_from(edges)
    return G


# Multiply Directed Graph for Wilson Cowan : clusters of nodes connected each other
def multiply_dir_graph(G, side, nodal=1, edges=1, other=0):
    A = nx.adjacency_matrix(G).todense()
    B = [1]*(len(A)*side)
    i = 0
    for r in A.tolist():
        row = [[]]*side
        j = 0
        for c in r:
            if c > 0:
                cluster = [[1 if col < edges else 0 for col in row] for row in np.random.rand(side, side)]
            elif j == i:
                cluster = [[1 if col < nodal else 0 for col in row] for row in np.random.rand(side, side)]
            else:
                cluster = [[1 if col < other else 0 for col in row] for row in np.random.rand(side, side)]
            row = np.hstack((row, cluster))
            j += 1
        B = np.vstack((B, row))
        i += 1
    B = B[1:][:]
    for i in range(len(B)):
                B[i][i] = 0
    return nx.DiGraph(B)


# generate Kuramoto model
def create_K(G,  # Graph
             start_amplitude=10,  # initial value of amplitudes
             int_freq=10,  # initial value for frequencies
             amp_cycle=10,  # amplitude cycle value for each node
             K=1):  # Coupling value for each node

    n = len(G)  # number of nodes
    # parameter initialization
    parameters = nm.create_vec_states([(n, int_freq, 2, 'normal'), (n, amp_cycle**2, 2, 'normal'), (n, K, .2, 'normal')])
    # initialization of states
    states = nm.create_vec_states([(n, 0, 2*np.pi, 'uniform'), (n, start_amplitude, 2, 'normal')])
    # convert polar to rectangular coordinates
    phase = np.array([np.cos(states[:, 0]) + 1j * np.sin(states[:, 0])])
    # create correct state matrix
    x = np.array([amp*theta for amp, theta in zip(states[:, 1], phase)]).T

    # curry node function to be unique to each node: each node has a unique intrinsic frequency and amplitude
    def curry_node(w, a):
        def node(x):   # node function
            return np.array([x[0] * (1j * w + a - abs(x[0])**2)])
        return node

    # w : intrinsic freq. a : limit cycle amplitude
    h = [curry_node(w, a) for w, a in zip(parameters[:, 0], parameters[:, 1])]

    # curry coupling function for each node for different K
    def curry_couple(K):
        def couple(x, y):  # coupling function
            return np.array([K/n*(y[0]-x[0])])
        return couple
    f = [curry_couple(K) for K in parameters[:, 2]]

    # Initialize model
    k_model = nm.nmodel(G, x, h, f, dt=.01)
    return k_model


# generate Wilson Cowan model (each node has E/I states)
def create_WC(G,
              CM,  # array containing tuple of strength of type of coupling
              # (0,0: e to e, 0,1: i to e, 1,0: e to i, 1,1: i to i)
              te=.1,  # time constant for excitatory
              ti=.065,  # time constant for inhibitory
              ke=1,  # max value of exc. sigmoid
              ki=1,  # max value of inh. sigmoid
              re=0,  # slope of linear exc. component
              ri=0,  # slope of linear inh. component
              c1=1,  # e to e constant
              c2=1.5,  # i to e constant
              c3=1,  # e to i constant
              c4=.25,  # i to i constant
              ae=50,  # exc. sigmoid parameter
              ai=50,  # inh. sigmoid parameter
              the=.125,  # threshold for excitatory
              thi=.5,  # threshold for inhibitory
              Ie=0,  # external input to excitatory
              Ii=0,  # external input to inhibitory
              e_i=.4,  # initial value for excitatory population
              i_i=.2):  # initial value for inhibitory population
    # extends network model class (changes dev function)
    class WC(nm.nmodel):
        def __init__(self, G, x, p, CM, g=None, dt=.05):
            super(WC, self).__init__(G, x, None, None, g, dt)
            self. p = p  # parameters
            self.CM = CM  # array determining connectivity of E/I

        def dev(self, state):
            # determines sum of contribution of neighborhood
            def edges(i, eori):
                tot = [0, 0]
                for j in range(len(G.edges())):
                    edge = G.edges()[j]
                    if edge[1] == i:
                        val = np.multiply(self.CM[j, eori, :], state[edge[0]])
                        tot += val
                return tot

            def S(x, a):  # sigmoid function
                return 1/(1 + np.exp(-a*(x)))
            dev = np.matrix(np.zeros(np.shape(state)))
            for i in range(len(state)):
                E = state[i][0]  # excitatory state for node i
                I = state[i][1]  # inhibitory state for node i
                pe = p[i, :, 0]  # excitatory parameters for node i
                pi = p[i, :, 1]  # inhibitory parameters for node i

                # differential equation
                dE = (-E + (pe[0] - pe[1]*E) * S((pe[2]*(E + edges(i, 0)[0]) -
                      pe[3]*(I + edges(i, 0)[1]) - pe[4] + pe[5]), pe[6]))/pe[7]
                dI = (-I + (pi[0] - pi[1]*I) * S((pi[2]*(E + edges(i, 1)[0]) -
                      pi[3]*(I + edges(i, 1)[1]) - pi[4] + pi[5]), pi[6]))/pi[7]

                dev[i] = [dE, dI]
            return dev

    n = len(G)  # number of nodes
    e = G.number_of_edges()  # number of edges

    # initialization of states
    x = nm.create_vec_states([(n, e_i), (n, i_i)])

    # parameters initialization
    parameters_e = nm.create_vec_states([(n, ke), (n, re), (n, c1), (n, c2), (n, the), (n, Ie), (n, ae), (n, te)])
    parameters_i = nm.create_vec_states([(n, ki), (n, ri), (n, c3), (n, c4), (n, thi), (n, Ii), (n, ai), (n, ti)])
    p = np.dstack((parameters_e, parameters_i))
    p[-1, -1, -1] = .075  # change time constant to fit Li's WC TwoPop model

    if len(CM) != e:
            raise ValueError('length of CM must match number of edges')

    return WC(G, x, p, CM, dt=.01)

# source = http://www.math.drexel.edu/~medvedev/classes/2006/math723/talks/Wilson_Cowan.pdf


# neural mass Jansen model
def create_NMM(G):
    pass
