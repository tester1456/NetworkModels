#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:08:02 2017

@authors: Rohit Konda, Vineet Tiruvadi
#Modules for working with dynamical network models
Notes:
    Might be a smart idea to merge with TVB modules if they already do a lot of this
"""

#import statements
import networkx as nx
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import collections as col


#Class  encapsulation of the network model 
class nmodel:
    def __init__(self, G, x, h, f, M, N, dt = .05):
        self.G = G # Graph representation of network
        self.x = y = np.array([np.array([np.array([j]) for j in i]) for i in np.array(x)]) # states 
        self.h = h # array of node functions
        self.f = f # array of coupling functions
        self.M = M # measurement matrix
        self.N = N # Variance for Gaussian noise
        self.y = np.array(self.linear_measure()) # measurement vectors
        self.t = 0 # time
        self.dt = dt # time step

        #checks
        if len(self.x) == 0:
            raise ValueError('self.x can\'t have less than one state')
        if len(self.x) != nx.number_of_nodes(self.G):
            raise ValueError('length of self.x must match number of nodes')
        if len(self.x) != len(self.M):
            raise ValueError('length of self.x must match number of nodes')

    #state derivative
    def dev(self,state):  
        if np.iscomplex(state).any():
            dev = np.matrix(np.zeros(np.shape(state),dtype=np.complex_))
        else:
            dev = np.matrix(np.zeros(np.shape(state)))
        c = 0 # counter if f depends on the edge
        for i in range(0,len(state)):
            sumEdge = np.array(dev[i].tolist()[0])
            if self.G[i+1]:
                for j in self.G[i+1]:
                    if isinstance(self.f, col.Iterable):
                        if len(self.f) == len(self.x):
                            sumEdge += self.f[i](state[i],state[j-1])
                        elif len(self.f) == self.G.number_of_edges():
                            sumEdge += self.f[c](state[i],state[j-1])
                            c += 1
                        else:
                            raise ValueError('length of f must either be equal to the number of nodes or edges')
                    elif callable(self.f):
                        sumEdge += self.f(state[i],state[j-1])
                    else:
                        raise ValueError('f must be either be iterable or callable')
            if isinstance(self.h, col.Iterable):
                dev[i] = self.h[i](state[i]) + sumEdge
            elif callable(self.h):
                dev[i] = self.h(state[i]) + sumEdge
            else:
                raise ValueError('h must be either be iterable or callable')
        return dev

    #linear measurement
    def linear_measure(self):  
        if self.N == 0:
            return self.M * np.matrix(self.x[:,:,-1])
        else:
            return self.M * np.matrix(self.x[:,:,-1]) + np.matrix(np.random.normal(0,self.N,np.shape(self.x[:,:,-1])))

    #euler method approximation of behavior
    def euler_step(self):
        new_state = self.x[:,:,-1] + self.dev(self.x[:,:,-1])*self.dt
        self.t += self.dt
        self.x = np.dstack((self.x,np.array(new_state)))
        self.y = np.dstack((self.y,np.array(self.linear_measure())))

    #runge-Kutta approximation of behavior
    def runge_kutta_step(self):
        k1 = self.dev(self.x[:,:,-1])*self.dt
        k2 = self.dev(np.array(self.x[:,:,-1]+ .5*k1))*self.dt
        k3 = self.dev(np.array(self.x[:,:,-1]+ .5*k2))*self.dt
        k4 = self.dev(np.array(self.x[:,:,-1]+ k3))*self.dt
        new_state = self.x[:,:,-1] + (k1+ 2*k2 + 2*k3 + k4)/6
        self.t += self.dt
        self.x = np.dstack((self.x,np.array(new_state)))
        self.y = np.dstack((self.y,np.array(self.linear_measure())))

    #time step function
    def step(self):
        self.runge_kutta_step()

    #runs model for time T and stores states
    def run(self,T):
        for ts in range(0,int(T/self.dt)):
            self.step()

    #clears all states exept initial
    def clear_run(self):
        self.x = np.array([np.array([np.array([int(j)]) for j in i]) for i in self.x[:,:,0]])
        self.y = self.y[:,:,0]


#lambda function for creating vectorized distributions
create_vec_states = lambda param: np.concatenate((np.array([create_states(*tup) for tup in param])),axis = 1)

#creates specified states
#n: number of states
#distribution: logistic, normal, uniform, or point
#for point: a = value
#for normal/logistic: a: mean, b: scale(spread)
#for uniform: a:lower bound, b:upper bound
#for complex tuple c: respective "a" and "b" for imaginary parts
def create_states(n, a, b = None, distribution = None, c = (0,0)):
    if distribution == None:
        if c == (0,0):
            return np.matrix([a]*n).T
        else:
            return np.matrix([a]*n).T + np.matrix([c[0]]*n).T * 1j
    elif distribution == 'logistic':
        if c == (0,0):
            return np.matrix(np.random.logistic(a,b,n)).T
        else:
            return np.matrix(np.random.logistic(a,b,n)).T + \
            np.matrix(np.random.logistic(c[0],c[1],n)).T * 1j
    elif distribution == 'normal':
        if c == (0,0):
            return np.matrix(np.random.normal(a,b,n)).T
        else:
            return np.matrix(np.random.normal(a,b,n)).T + \
            np.matrix(np.random.normal(c[0],c[1],n)).T * 1j
    elif distribution == 'uniform':
        if c == (0,0):
            return np.matrix(np.random.uniform(a,b,n)).T
        else:
            return np.matrix(np.random.uniform(a,b,n)).T + \
            np.matrix(np.random.uniform(c[0],c[1],n)).T * 1j

#creates specified network: K, SW, BA, SCC(6 nodes)
#n: number of nodes
#k: each node is connected to k nearest neighbors in ring topology
#p: the probability of rewiring each edge
#m: number of edges to attach from a new node to existing nodes
def create_network(ntype = '', n = 1, k = 1, p = 1, m = 1):
    if ntype == 'K':
        return nx.complete_graph(n)
    elif ntype == 'SW':
        return nx.watts_strogatz_graph(n, k, p)
    elif ntype == 'BA':
        return nx.barabasi_albert_graph(n, m)
    elif ntype == 'SCC':
        nodes = [1,2,3,4,5,6]
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from([(1,2),(2,3),(1,6),(1,3),(3,4),(4,5),(4,6),(5,6)])
        return G
    else:
        raise ValueError('not a valid network')

#multiply the input graph by a factor of side
# nodal: probability of connectivity in nodes
# edges: probability of connectivity within edges
# other: probability of connectivity with no edges
def multiply_graph(G, side, nodal = .9, edges = .5, other = .1):
    A = nx.to_numpy_matrix(G)
    B = [1]*(len(A)*side)
    i = 0
    for r in A.tolist():
        row = [[]]*side
        j = 0
        for c in r:
            if c > 0:
                cluster = [[1 if col < edges else 0 for col in row] for row in np.random.rand(side,side)]
            elif j == i:
                cluster = [[1 if col < nodal else 0 for col in row] for row in np.random.rand(side,side)]
            else:
                cluster = [[1 if col < other else 0 for col in row] for row in np.random.rand(side,side)]
            row = np.hstack((row,cluster))
            j += 1
        B = np.vstack((B,row))
        i += 1
    B = B[1:][:]
    for i in range(len(B)):
                B[i][i] = 0
                for j in range(i):
                    B[i][j] = B[j][i]
    return nx.from_numpy_matrix(B)

#plots connections as an adjacency matrix
def plt_graph(G):
    A = nx.to_numpy_matrix(G)
    plt.figure()
    plt.imshow(A, interpolation= "nearest")

#plots states
def state_course(states):
    plt.figure()
    plt.plot(states.T)
    plt.xlabel('Time Steps')
    plt.ylabel('x')
    plt.title('t')
    plt.show()

#plots spectogram
def spectrogram(signal, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=None, nfft=None):
    f, t, s = sig.spectrogram(signal, fs, window, nperseg, noverlap, nfft)
    plt.pcolormesh(t, f, s)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()

#plots PSD
def PSD(signal, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None):
    f, Pxx = sig.welch(signal, fs, window, nperseg, noverlap, nfft)
    plt.semilogy(f, Pxx)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()

#runs functions across all signals and outputs matrix of results
def cross_func(states, func):
    M = np.zeros((len(states), len(states)))
    for i in range(len(states)):
        for j in range(len(states)):
            M[i,j] = func(states[i,:],states[j,:])
    return np.matrix(M)

#Source: https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/metrics/cluster/supervised.py#L531
#Mutual Information
def mutual_info_score(labels_true, labels_pred, contingency=None):
    if contingency is None:
        labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    else:
        contingency = check_array(contingency,
                                  accept_sparse=['csr', 'csc', 'coo'],
                                  dtype=[int, np.int32, np.int64])

    if isinstance(contingency, np.ndarray):
        # For an array
        nzx, nzy = np.nonzero(contingency)
        nz_val = contingency[nzx, nzy]
    elif sp.issparse(contingency):
        # For a sparse matrix
        nzx, nzy, nz_val = sp.find(contingency)
    else:
        raise ValueError("Unsupported type for 'contingency': %s" %
                         type(contingency))

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx) * pj.take(nzy)
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (contingency_nm * (log_contingency_nm - log(contingency_sum)) +
          contingency_nm * log_outer)
    return mi.sum()

#PLV
def phase_locking_value(x,y):
    complex_phase_diff = np.exp(np.complex(0,1)*(x - y))
    plv = np.abs(np.sum(complex_phase_diff))/len(x)
    return plv

#coherence
coh = lambda x,y: np.mean(sig.coherence(x,y)[1])

#correlation
cor = np.correlate

#get time courses from specified part of the vector state
reduce_state = lambda i, x:  np.matrix([row[i] for row in x])

#Nonlinear measures
#import nolds.sampen as as sp
#import nolds.corr_dim as cd
#import nolds.lyap_r as lp

#Granger Causality
#import nitime.analysis.granger as GC