import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import scipy.signal as sig
import scipy.stats as stats
import statsmodels
import network_modules as nm
import graph_measures as gm
import pickle

#interfacing with measures
list_of_measures = []

u_p_s = lambda X: gm.cross_func(X,gm.phase_synchrony)
list_of_measures.append(u_p_s)

u_c = lambda X: gm.cross_func(X,gm.correlation)
p_c = lambda X: gm.partial_method(X,gm.correlation)
list_of_measures.append(u_c)
list_of_measures.append(p_c)

u_ch = lambda X: gm.cross_func(np.real(X),gm.coherence)
p_ch = lambda X: gm.partial_method(np.real(X),gm.coherence)
list_of_measures.append(u_ch)
list_of_measures.append(p_ch)

u_mi = lambda X: gm.cross_func(np.real(X),gm.kraskov_mi)
p_mi = lambda X: gm.partial_method(np.real(X),gm.kraskov_mi)
list_of_measures.append(u_mi)
list_of_measures.append(p_mi)


u_te = lambda X: gm.cross_func(np.real(X),gm.kernel_TE)  #
p_te = lambda X: gm.partial_method(np.real(X),gm.kernel_TE)  #
list_of_measures.append(u_te)
list_of_measures.append(p_te)


u_gc = lambda X: gm.cross_func(np.real(X),gm.granger_causality)  #
p_gc = lambda X: gm.partial_method(np.real(X),gm.granger_causality)  #
list_of_measures.append(u_gc)
list_of_measures.append(p_gc)

u_r2 = lambda X: gm.cross_func(np.real(X),gm.r2)
p_r2 = lambda X: gm.partial_method(np.real(X),gm.r2)
list_of_measures.append(u_r2)
list_of_measures.append(p_r2)

u_n2 = lambda X: gm.cross_func(np.real(X),gm.n2)
p_n2 = lambda X: gm.partial_method(np.real(X),gm.n2)
list_of_measures.append(u_n2)
list_of_measures.append(p_n2)

def PDC_func(X):  #
    X = np.real(X) 
    p = 2
    A = gm.MVAR_fit(X,p)
    P, freq = gm.PDC(A)
    P_mat = np.amax(P,0)
    np.fill_diagonal(P_mat,0)
    return P_mat
list_of_measures.append(PDC_func)

def DTF_func(X):  #
    X = np.real(X) 
    p = 2
    A = gm.MVAR_fit(X,p)
    D, freq = gm.DTF(A)
    D_mat = np.amax(D,0)
    np.fill_diagonal(D_mat,0)
    return D_mat
list_of_measures.append(DTF_func)


freqForAmp = 1.5*np.arange(1,61)
freqForPhase = np.arange(1,61)/4+1
sr = 1000
bw = 1.5

KBL = lambda X,Y: sum(sum(gm.KLDivMIcomod(X,Y,freqForAmp,freqForPhase,sr,bw)))  #
d_KBL = lambda X: gm.dir_cross_func(np.real(X), KBL)
list_of_measures.append(d_KBL)

MV = lambda X,Y: sum(sum(gm.zScoreMVcomod(X,Y,freqForAmp,freqForPhase,sr,bw)[0]))  #
d_MV = lambda X: gm.dir_cross_func(np.real(X), MV)
list_of_measures.append(d_MV)

PLV = lambda X,Y: sum(sum(gm.PLVcomod(X,Y,freqForAmp,freqForPhase,sr,bw)))  #
d_PLV = lambda X: gm.dir_cross_func(np.real(X), PLV)
list_of_measures.append(d_PLV)

GLM = lambda X,Y: sum(sum(gm.GLMcomod(X,Y,freqForAmp,freqForPhase,sr,bw)))  #
d_GLM = lambda X: gm.dir_cross_func(np.real(X), GLM)
list_of_measures.append(d_GLM)

#threshold classifier for adjacency matrices, func_data is an array of connectivity arrays
def gen_threshold(adj_data, func_data, greater_than = True):
    adj_arr = adj_data.flatten()
    func_arr = func_data.flatten()
    thr = 0
    m = 0
    for x in func_arr:
        if greater_than:
            thr_arr = 1*(func_arr >= x)
        else:
            thr_arr = 1*(func_arr <= x)
        d = sum(1*(thr_arr == adj_arr))
        if d > m:
            m = d
            thr = x
    return thr


# apply threshold t to array
def thresh_array(arr, t, greater_than=True):
    if greater_than:
        thr_arr = 1*(arr >= t)
    else:
        thr_arr = 1*(arr <= t)
    return thr_arr

# Noise sources
white_noise = lambda X,dev: X + np.random.normal(0, dev, X.shape) + 1j*np.random.normal(0, dev, X.shape)  #

downsample = lambda X,step: X[:, 0:-1:step]  #

shorten = lambda X, l: X[:, 0:l]  #

electrode_inv = lambda X , A: A.T * np.linalg.inv(A*A.T) * A * X  #

minus_reference = lambda X: X - np.mean(X, 0)  #

time_delay = lambda X,delays: np.array([node[d:(len(X[0]) - max(delays) + d)] for node, d in zip(X, delays)])  #


def colored_noise(X, func, dev):
    def get_noise():
        N = len(X[0])
        f = np.array([x + 1 for x in range(N)]) / 4
        mag = np.sqrt(func(f))
        phase = 2*np.pi*np.random.random(N)
        FFT = mag * np.exp(1j*phase)
        tsig = np.fft.ifft(FFT)
        tsig = (tsig - np.mean(tsig)) / max(abs(tsig)) * 2 * dev
        return tsig
    return np.array([sig + get_noise() for sig in X])
    
def generate_data():
    # run n monte carlo simulations for time t get adjacency and time data
    k_data = []
    wc_data = []
    o_data = []
    adj_data = []
    n = 100
    for i in range(n):
        nodes = 25
        G = generate_graph(nodes)
        adj_data.append(nx.to_numpy_matrix(G))

        kmodel = generate_k_model(nodes, G)
        wcmodel = generate_wc_model(nodes, G)
        omodel = generate_o_model(nodes, G)

        t = 100
        kmodel.run(t)
        k_data.append(kmodel.x)
        wcmodel.run(t)
        wc_data.append(wcmodel.x)
        omodel.run(t)
        o_data.append(omodel.x)

    with open('k_data', 'wb') as f:
        pickle.dump(k_data, f)
    with open('wc_data', 'wb') as f:
        pickle.dump(wc_data, f)
    with open('o_data', 'wb') as f:
        pickle.dump(o_data, f)

    with open('adj_data', 'wb') as f:
        pickle.dump(adj_data, f)

    print('Models Generated')


def generate_noisy_data(data):
    with open('k_data', 'rb') as pickle_data:
        trial_data = pickle.load(pickle_data)


white_noise
downsample
shorten
electrode_inv
minus_reference
time_delay
colored_noise

'''
    # generate connectivity matrixes for each measure for normal
    with open('time_data' , 'rb') as pickle_data:
        trial_data = np.array(pickle.load(pickle_data))
        adj_data = []
        for trial in trial_data:
            trial_adj = []
            for func in list_of_measures:
                trial_adj.append(func(trial[:,0,:]))
                print(func.__name__)
            adj_data.append(trial_adj)
            print('Trial Done')
        with open('measure_normal', 'wb') as f:
            f.write(pickle.dumps(adj_data))

        print('Measures conducted')
'''

if __name__ == '__main__':

    print('Running main')

    generate_data()

    with open('k_data', 'rb') as pickle_data:
        trial_data = pickle.load(pickle_data)
        generate_noisy_data(trial_data)
    with open('wc_data', 'rb') as pickle_data:
        trial_data = pickle.load(pickle_data)
        generate_noisy_data(trial_data)
    with open('o_data', 'rb') as pickle_data:
        trial_data = pickle.load(pickle_data)
        generate_noisy_data(trial_data)
