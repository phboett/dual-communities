#!usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains methods to simulate a stochastic model in the spirit
of Corson, Fluctuations and Redundancy in Optimal Transport Networks (https://doi.org/10.1103/PhysRevLett.104.048703),
but with multiple sources that fluctuate additionally"""

import networkx as nx
import numpy as np


def build_artificial_leaf_triangular_multisource_dirichlet(N: int,
                                                           gamma: float,
                                                           mu: float = -1.,
                                                           sigma: float=.1,
                                                           cap_K: float=1,
                                                           threshold: float = 1e-6,
                                                           nof_sources: int=3,
                                                           dirichlet_parameter: float = 0.,
                                                           dirichlet_constant: float = 500.,
                                                           max_repetitions: int = 1000) -> nx.Graph:
    """Implement the model by Corson, 2010, Fluctuations and Redundancy in Optimal Transport Networks, PRL
    which yields random graphs in between trees and square grids depending on the parameter gamma \\in [0,infty]
    In contrast to the original model, only the sinks are still iid uncorrelated Gaussian variables with
    mean mu and standard deviation sigma, whereas the sources have an additional, additive noise given by
    dirichlet random variables. They have a multiplicative factor given by the Dirichlet constant and a parameter
    that can be tuned from dirichlet_parameter-> 0 for almost always all sources having the same additive random
    variable and dirichlet_parameter-> infty for only one source being activated in each realization.

    Args:
        N (int): Number of edge nodes.
        gamma (float): Parameters tuning between random networks.
        mu (int, optional): Mean of gaussian distribution. Defaults to -1.
        sigma (int, optional): Parameter of gaussian dist. Defaults to .1
        cap_K (int, optional): _description_. Defaults to 1.
        threshold (_type_, optional): Threshold to stop the optimization to find network. Defaults to 1e-6.
        nof_sources (int, optional): Number of sources. Can only take 2,3 or 6 due to geometrical constraints.
        Defaults to 3.
        dirichlet_parameter (int, optional): Parameter of the dirichlet distribution
        which is referred to as 'alpha' in the publication.
        Defaults to 0.
        dirichlet_constant (int, optional): Scale constant which has the symbol 'K' in the publication.
        Defaults to 500.
        max_repetitions (int, optional): _description_. Defaults to 1000.

    Returns:
        G (networkx.Graph): Network with optimized weights.
    """
    
    if nof_sources not in [2,3,6]:
        raise NotImplementedError('Only supports 2,3,6 sources!')

    G = nx.generators.triangular_lattice_graph(N,2*N)
    
    for i in range(0,int(N/2)-1):
        G.remove_nodes_from([(i,N/2-2*i-1),(i,N/2-2*i-2)])
        G.remove_nodes_from([(i,N/2+2*i+1),(i,N/2+2*i+2)])
        
    for j in range(0,int(N/2)-2):
        G.remove_nodes_from([(N-j,N/2-2*j-2),(N-j,N/2-2*j-3)])
        G.remove_nodes_from([(N-j,N/2+2*j+2),(N-j,N/2+2*j+3)])
        
    main_comp = np.max([len(comp) for comp in list(nx.connected_components(G))])
    
    for comp in list(nx.connected_components(G)):
        if len(comp)!=main_comp:
            G.remove_nodes_from(comp)

    nof_nodes = len(list(G.nodes()))

    if nof_sources ==2:
        ind = list(G.nodes()).index((0,int(N/2)))
        ind2 = list(G.nodes()).index((N,int(N/2)))
        indices = [ind,ind2]

    elif nof_sources == 3:
        ind = list(G.nodes()).index((0,int(N/2)))
        ind2 = list(G.nodes()).index((int(3*N/4)+1,int(N)))
        ind3 = list(G.nodes()).index((int(3*N/4)+1,0))
        indices = [ind,ind2,ind3]

    elif nof_sources == 6:
        ind = list(G.nodes()).index((0,int(N/2)))
        ind2 = list(G.nodes()).index((N,int(N/2)))
        ind3 = list(G.nodes()).index((int(3*N/4)+1,int(N)))
        ind4 = list(G.nodes()).index((int(3*N/4)+1,0))
        ind5 = list(G.nodes()).index((int(N/4)+1,int(N)))
        ind6 = list(G.nodes()).index((int(N/4)+1,0))
        indices = [ind,ind2,ind3,ind4,ind5,ind6]

    correlation_matrix_sources = np.ones((nof_nodes,nof_nodes))*mu**2
    np.fill_diagonal(correlation_matrix_sources,mu**2+sigma**2)
    mask = np.ones(nof_nodes)
    mask[indices] = 0
    mask = mask.astype(bool)
    mus = np.array([-mu*(nof_nodes-nof_sources)/nof_sources for i in range(nof_sources)])
    sigmas = np.sqrt(np.array([sigma**2*(nof_nodes-nof_sources)/nof_sources for i in range(nof_sources)]))

    covariance_dirichlets = np.zeros((nof_sources,nof_sources))
    for i in range(nof_sources):
        for j in range(nof_sources):
            if i == j:
                covariance_dirichlets[i,j] = (dirichlet_constant**2*(nof_sources-1)/
                                              (nof_sources**2*(nof_sources*dirichlet_parameter+1)))
            else:
                covariance_dirichlets[i,j] = (-dirichlet_constant**2*1/
                                              (nof_sources**2*(nof_sources*dirichlet_parameter+1)))
    #### correlation matrix for sources
    count1 = 0
    for index in indices:
        correlation_matrix_sources[index,mask] = - ((nof_nodes-nof_sources)*mu**2+sigma**2)/nof_sources
        correlation_matrix_sources[mask,index] = - ((nof_nodes-nof_sources)*mu**2+sigma**2)/nof_sources
        count2 = 0
        for index2 in indices:
            correlation_matrix_sources[index,index2] = mus[count1]*mus[count2] + covariance_dirichlets[count1,count2]
            if index2 == index:
                correlation_matrix_sources[index,index2] += sigmas[count2]**2
            count2+=1
        count1+=1
    capacities = np.random.random(len(G.edges()))
    capacities /= (np.sum(capacities**gamma))**(1/gamma)/(cap_K**gamma)
    capacity_dict = {}
    for i in range(len(G.edges())):
        capacity_dict[list(G.edges())[i]] = capacities[i]

    nx.set_edge_attributes(G,capacity_dict,'weight')
    last_change = 1e5
    iterations = 0
    while last_change > threshold:

        if iterations % 10 == 0:
            print('Iterations {0}, Last Change {1:.4E}'.format(iterations, last_change), end='\r')
            
        line_capacities = np.array([G[u][v]['weight'] for u,v in G.edges()])
        L = nx.laplacian_matrix(G).todense()
        B = np.linalg.pinv(L)
        I = nx.incidence_matrix(G,oriented=True).todense()
        flow_correlations = np.linalg.multi_dot([np.diag(line_capacities), I.T, B,correlation_matrix_sources, 
                                                 B, I, np.diag(line_capacities)])
        #flow_correlations = calc_diagonal_flow_correlations(G,correlation_matrix_sources)
        new_line_capacities = np.zeros(len(G.edges()))
        for i in range(len(flow_correlations)):
            new_line_capacities[i] = (flow_correlations[i,i]**(1/(1+gamma))/
                                      (np.sum(np.diag(flow_correlations)**(gamma/(1+gamma))))**(1/gamma)*cap_K)
            
        new_cap_dict = {list(G.edges())[i]:new_line_capacities[i] for i in range(len(G.edges()))}
        nx.set_edge_attributes(G,new_cap_dict,'weight')
        last_change = np.sum((new_line_capacities-line_capacities)**2)
        iterations += 1
        if iterations > max_repetitions:
            print('Maximum number of repetitions reached!')
            break
        
    return G