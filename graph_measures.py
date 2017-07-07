import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import scipy.signal as sig
import statsmodels

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
    A = A.tolist()
    B = B.tolist()
    andsum = sum([sum([i and j for i,j in zip(a,b)]) for a,b in zip(A,B)])
    orsum = sum([sum([i or j for i,j in zip(a,b)]) for a,b in zip(A,B)])
    return andsum/orsum

#graph similarity index
def cosine(A,B):
    A = A.tolist()
    B = B.tolist()
    mag = lambda x: np.sqrt(np.dot(x,x))
    return 1 - (sum([1 - (np.dot(a,b)/(mag(a)*mag(b))) for a,b in zip(A,B)])/len(A))

#graph similarity index
def distance(A,B):
    dis = sum([sum(x) for x in abs(A-B).tolist()])
    return 1 - (dis/len(A))**2

#applies function to all pairs of nodes
def cross_func(states, func):
    M = np.zeros((len(states), len(states)))
    for i in range(len(states)):
        for j in range(len(states)):
            if i != j:
                M[i,j] = func(states[i,:],states[j,:])
    return np.matrix(M)

#phase synchrony measure
def phase_synchrony(x,y):
    ps = np.mean(abs((x+y)))/2
    return ps

#correlation
cor = lambda x,y: np.correlate(np.array(x)[0],np.array(y)[0])[0]/(len(np.array(x)[0]))

#coherence
coh = lambda x,y: max(sig.coherence(np.array(x)[0],np.array(y)[0])[1])

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

#Nonlinear measures
#import nolds.sampen as as sp
#import nolds.corr_dim as cd
#import nolds.lyap_r as lp