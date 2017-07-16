import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import scipy.signal as sig
import scipy.stats as stats
import statsmodels
import scipy.linalg as lin
import math
from scipy.fftpack import fft,fftfreq


#plot degree distribution of graph
def degree_distribution(G):    
    degree_sequence=sorted([d for n,d in G.degree().items()], reverse=True) # degree sequence
    degreeCount=collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.plot(deg, cnt,'.')

    plt.title("Degree Distribution")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d+0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.show()

#graph similiarity index
def jaccard(A, B):
    andsum = sum([sum([i and j for i,j in zip(a,b)]) for a,b in zip(A,B)])
    orsum = sum([sum([i or j for i,j in zip(a,b)]) for a,b in zip(A,B)])
    if orsum == 0:
        return 1.0
    else: 
        return andsum/orsum

#graph similarity index
def cosine(A,B):
    import itertools
    mag = lambda x: np.sqrt(np.dot(x,x))
    test_z = lambda arr: all(x == 0 for x in itertools.chain(*arr))
    if test_z(A) or test_z(B):
        if test_z(A) and test_z(B):
            return 1.0
        else:
            return 0.0
    else:
        return 1 - (sum(np.nan_to_num([1 - (np.dot(a,b)/(mag(a)*mag(b))) for a,b in zip(A,B)]))/len(A))

#graph similarity index
def distance(A,B):
    dis = sum([sum(x) for x in abs(A-B)])
    n = len(A)
    return max((n**2 - dis - n)/n**2 * n/(n-1) , 0)

#applies function to all pairs of nodes
def cross_func(states, func):
    M = np.zeros((len(states), len(states)))
    for i in range(len(states)):
        M[i,i] = 0
        for j in range(i+1,len(states)):
            val = func(states[i,:],states[j,:])
            M[i,j] = val
            M[j,i] = val
    return np.matrix(M)

#phase synchrony measure
def phase_synchrony(X,Y):
    ps = np.mean(abs((X+Y)))/2
    return ps

#correlation
correlation = lambda X,Y: np.correlate(np.array(X)[0],np.array(Y)[0])[0]/(len(np.array(X)[0]))

#coherence
coherence = lambda X,Y: max(sig.coherence(np.array(X)[0],np.array(Y)[0])[1])

#mutual information measure using KNN Kraskov 2004
def kraskov_mi(X, Y, k, est = 1):
    from scipy.special import digamma
    
    if len(X[0]) != len(Y[0]):
        raise ValueError('length of X and Y must match')
    
    n = len(X[0])
    dx = np.zeros((n,n))
    dy = np.zeros((n,n))
    dz = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            dx[i,j] = np.sqrt(sum(X[:,i] - X[:,j])**2)
            dy[i,j] = np.sqrt(sum(Y[:,i] - Y[:,j])**2)
            dz[i,j] = max(dx[i,j],dy[i,j])
    
    nx = np.zeros((n,1))
    ny = np.zeros((n,1))
    
    for i in range(n):
        dxi = dx[i,:]
        dyi = dy[i,:]
        dzi = dz[i,:]
        
        epsi = np.sort(dzi)
        
        if est == 1:
            nx[i] = sum([1 for x in dxi if x < epsi[k]])
            ny[i] = sum([1 for y in dyi if y < epsi[k]])
        elif est == 2: 
            nx[i] = sum([1 for x in dxi if x <= epsi[k]])
            ny[i] = sum([1 for y in dyi if y <= epsi[k]])
        else:
            raise ValueError('est must be either 1 or 2')
    
    if est == 1:
        return digamma(k) - sum(digamma(nx) + digamma(ny))/n + digamma(n)
    else:
        return digamma(k) - 1/k - sum(digamma(nx-1) + digamma(ny-1))/n + digamma(n)

#transfer entropy of X onto Y
def kernel_TE(X, Y):
    xn = X[:-1]
    yn = Y[:-1]
    yn1 = Y[1:]
    n = len(xn)
    
    gauss_ker = lambda x,o: np.exp(-((x/o)**2)/2)/(o*np.sqrt(2*np.pi))
    oxn = 1.06*np.std(xn)*n**.2
    oyn = 1.06*np.std(yn)*n**.2
    oyn1 = 1.06*np.std(yn1)*n**.2
    
    TE = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pxnynyn1 = sum(gauss_ker(xn-xn[i],oxn)*gauss_ker(yn-yn[j],oyn)*gauss_ker(yn1-yn1[k],oyn1))/n
                pxnyn = sum(gauss_ker(xn-xn[i],oxn)*gauss_ker(yn-yn[j],oyn))/n
                pynyn1 = sum(gauss_ker(yn-yn[j],oyn)*gauss_ker(yn1-yn1[k],oyn1))/n
                pyn = sum(gauss_ker(yn-yn[j],oyn))/n
                TE += pxnynyn1 * np.log2(pxnynyn1*pyn/(pxnyn*pynyn1))
    return TE

#granger causality of X onto Y
granger_causality = lambda X,Y,ml: statsmodels.tsa.stattools.grangercausalitytests(np.stack((Y,X),1), maxlag = ml, verbose = False)

#pearson correlation coefficient
coeff_determination = lambda X,Y: stats.pearsonr(X,Y)[0]**2

#correlation ratio
def correlation_ratio(X,Y):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    mean = (mean_x + mean_y)/2
    cat_disp = len(X)*(mean_x - mean)**2 + len(Y)*(mean_y - mean)**2
    disp_x = sum([(x - mean)**2 for x in X])
    disp_y = sum([(y - mean)**2 for y in Y])
    return cat_disp/(disp_x+disp_y)

#nonlinear correlation ratio based on kernel estimate of function
def h2(X,Y):
    gauss_ker = lambda x,o: np.exp(-((x/o)**2)/2)/(o*np.sqrt(2*np.pi))
    mean_y = np.mean(Y)
    SS_y = sum([(y - mean_y)**2 for y in Y])
    o = 1.06*np.std(Y)*len(Y)**.2
    plt.plot(X,[sum([Y[i]*gauss_ker(X-x,o)[i] for i in range(len(Y))])/sum(gauss_ker(X-x,o)) for x in X])
    SS_res = sum([(y - sum([Y[i]*gauss_ker(X-x,o)[i] for i in range(len(Y))]))**2 for x,y in zip(X,Y)])
    h2 = 1 - SS_res/SS_y
    return h2

#partial based methods
def partial_method(X, method):
    nvar = len(X)
    M = np.zeros((nvar, nvar))
    for i in range(nvar):
        M[i, i] = 0
        for j in range(i+1, nvar):
            
            A = np.transpose(np.delete(X,[i,j],0))
            w_i = lin.lstsq(A,X[i])[0]
            w_j = lin.lstsq(A, X[j])[0]
            
            r_j = X[i] - np.dot(A,w_i)
            r_i = X[j] - np.dot(A,w_j)
            val = method(r_i, r_j)
            
            #test for nan
            if val == val:
                M[i, j] = val
                M[j, i] = val
            else:
                M[i, j] = 0
                M[j, i] = 0  
    return M

#return coefficients for MVAR model
def MVAR_fit(X,p):
    v, n = np.shape(X)
    
    cov = np.zeros((p+1, v, v))
    for i in range(p+1):
        cov[i] = np.cov(X[:,0:n-p],X[:,i:n-p+i])[v:,v:]
        
    G = np.zeros((p*v,p*v))
    for i in range(p):
        for j in range(p):
            G[v*i:v*(i+1) , v*j:v*(j+1)] = cov[abs(j-i)]
    
    cov_list = np.concatenate(cov[1:],axis=0)
    phi = linalg.lstsq(G, cov_list)[0]
    phi = np.reshape(phi,(p,v,v))
    for k in range(p):
        phi[k] = phi[k].T
    return phi

#spectral density helper function
def spectral_density(A, n_fft=None):
    p, N, N = A.shape
    if n_fft is None:
        n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)
    A2 = np.zeros((n_fft, N, N))
    A2[1:p + 1, :, :] = A  # start at 1 !
    fA = fft(A2, axis=0)
    freqs = fftfreq(n_fft)
    I = np.eye(N)

    for i in range(n_fft):
        fA[i] = lin.inv(I - fA[i])

    return fA, freqs

#directed transfer function
def DTF(A, sigma=None, n_fft=None):
    p, N, N = A.shape

    if n_fft is None:
        n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)

    H, freqs = spectral_density(A, n_fft)
    D = np.zeros((n_fft, N, N))

    if sigma is None:
        sigma = np.ones(N)

    for i in range(n_fft):
        S = H[i]
        V = (S * sigma[None, :]).dot(S.T.conj())
        V = np.abs(np.diag(V))
        D[i] = np.abs(S * np.sqrt(sigma[None, :])) / np.sqrt(V)[:, None]

    return D, freqs

#partial directed coherence
def PDC(A, sigma=None, n_fft=None):
    p, N, N = A.shape

    if n_fft is None:
        n_fft = max(int(2 ** math.ceil(np.log2(p))), 512)

    H, freqs = spectral_density(A, n_fft)
    P = np.zeros((n_fft, N, N))

    if sigma is None:
        sigma = np.ones(N)

    for i in range(n_fft):
        B = H[i]
        B = linalg.inv(B)
        V = np.abs(np.dot(B.T.conj(), B * (1. / sigma[:, None])))
        V = np.diag(V)  # denominator squared
        P[i] = np.abs(B * (1. / np.sqrt(sigma))[None, :]) / np.sqrt(V)[None, :]

    return P, freqs