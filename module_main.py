import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import scipy.signal as sig
import scipy.stats as stats
import statsmodels
import network_modules as nm
import graph_measures as gm
import json


#model generator
def generate_model():
	oscillator = nm.nmodel(G,x,h,f,N)
	return oscillator 

#run n monte carlo simulations for time t get adjacency and time series data
time_data = []
adj_data = []
n = 1000
for i in range(n):
	oscillator  = generate_model()
	t = 100
	oscillator.run(t)
	time_data.append(oscillator.x)
	adj_data.append(nx.to_numpy_matrix(oscillator.G))
with open('time_data' ,'w') as f:
    f.write(json.dumps(time_data))
with open('adj_data', 'w') as f:
	f.write(json.dumps(adj_data))

#interfacing with measures
list_of_measures = []

u_p_s = lambda X: gm.cross_func(X,gm.phase_synchrony)
p_p_s= lambda X: gm.partial_method(X,gm.phase_synchrony)
list_of_measures.append(u_p_s)
list_of_measures.append(p_p_s)

u_c = lambda X: gm.cross_func(X,gm.correlation)
p_c = lambda X: gm.partial_method(X,gm.correlation)
list_of_measures.append(u_c)
list_of_measures.append(p_c)

u_ch = lambda X: gm.cross_func(X,gm.coherence)
p_ch = lambda X: gm.partial_method(X,gm.coherence)
list_of_measures.append(u_ch)
list_of_measures.append(p_ch)

u_mi = lambda X: gm.cross_func(X,gm.kraskov_mi)
p_mi = lambda X: gm.partial_method(X,gm.kraskov_mi)
list_of_measures.append(u_mi)
list_of_measures.append(p_mi)

u_mi = lambda X: gm.cross_func(X,gm.kraskov_mi)
p_mi = lambda X: gm.partial_method(X,gm.kraskov_mi)
list_of_measures.append(u_mi)
list_of_measures.append(p_mi)


u_gc = lambda X: gm.cross_func(X,gm.granger_causality)
p_gc = lambda X: gm.partial_method(X,gm.granger_causality)
list_of_measures.append(u_gc)
list_of_measures.append(p_gc)

u_r2 = lambda X: gm.cross_func(X,gm.r2)
p_r2 = lambda X: gm.partial_method(X,gm.r2)
list_of_measures.append(u_r2)
list_of_measures.append(p_r2)

u_n2 = lambda X: gm.cross_func(X,gm.n2)
p_n2 = lambda X: gm.partial_method(X,gm.n2)
list_of_measures.append(u_n2)
list_of_measures.append(p_n2)

def PDC_func(X): 
    p = 2
    A = gm.MVAR_fit(X,p)
    P, freq = gm.PDC(A)
    P_mat = np.amax(P,0)
    np.fill_diagonal(P_mat,0)
    return P_mat
list_of_measures.append(PDC_func)

def DTF_func(X): 
	p = 2
	A = gm.MVAR_fit(X,p)
	D, freq = gm.DTF(A)
	D_mat = np.amax(D,0)
    np.fill_diagonal(D_mat,0)
    return D_mat
list_of_measures.append(DTF_func)

#generate connectivity matrixes for each measure for normal
with open('time_data') as json_data:
    trial_data = np.array(json.load(json_data))
    adj_data = []
    for trial in trial_data
    trial_adj = []
    for func in list_of_measures:
    	trial_adj.append(func(trial))
    adj_data.append(trial_adj)
    with open('measure_normal', 'w') as f:
    	 f.write(json.dumps(adj_data))
