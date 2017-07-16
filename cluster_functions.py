import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
import scipy.signal as sig
import scipy.stats as stats
import statsmodels
import network_modules as nm
import graph_measures as gm
import func_convolutions as fc
from json import dumps


#tests functions 
def test_func():
	pass

#run n monte carlo simulations
def gen_data(func_conv_list, n):
	data_list = []
	for func in func_conv_list:
		network_models = [func() for i in range(n)]
		data_list.append((func.__name__,network_models))
	return data_list

#run connectivity measures
def run_measures(data_list,list_measures):
	A_from_measures = []
	for data in data_list:
		A_from_measures.append([func(data) for func in list_measures])
	return A_from_measures

#compare adjacency matrixes from measured and actual
def compare_adjacency(A_list,A_from_measures):
	n, m = np.shape(A_from_measures)
	compare_list = []
	for i in range((len(A_list))):
		compare_list.append([(gm.cosine(A_list[i],B),gm.distance(A_list[i],B)) for B in A_from_measures[i]])
	return compare_list


with open('measure_data' ,'w') as f:
	write = lambda str: f.write(string + '\n')