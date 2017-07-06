import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import scipy.signal as sig

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

#Nonlinear measures
#import nolds.sampen as as sp
#import nolds.corr_dim as cd
#import nolds.lyap_r as lp
#Granger Causality
#import nitime.analysis.granger as GC