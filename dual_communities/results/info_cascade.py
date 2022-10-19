#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Simulation for cascade on primal and dual communities."""


import numpy as np
import networkx as nx

import gzip
import pickle

import time
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial

from dual_communities.dual_graph import create_dual_from_graph
from dual_communities.tools import assign_pos

out_data_path = "generated_data"


def edge_community_to_node_community(G, ecomm):
    """Convert the list of edge indices to a list of nodes and check if nodes are tuples or numbers
    NOTE: This function assumes that nodes in G are either tuples or numbers."""

    nodes_are_tuples = False
    if isinstance(list(G.nodes())[0],tuple):
        nodes_are_tuples = True

    fl = [[list(G.edges())[i][0],list(G.edges())[i][1]] for i in ecomm]
    flat = [item for sublist in fl for item in sublist]
    flat_temp = np.unique(flat,axis=0)
    if nodes_are_tuples:
        ncomm = [tuple(x) for x in flat_temp]
    else:
        ncomm = [x for x in flat_temp]
    ncomm_index = [list(G.nodes()).index(n) for n in ncomm]
    return ncomm,ncomm_index


def generate_randomised_lattice(N: int, p: float, w: float, MM=None,
                                strengthened_region: bool = False,
                                seed=None) -> nx.Graph:
    """Create a lattice where the edges lying in the central strengthened region get
    weight w with probability p.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if MM is None:
        M = 2 * N + 1
        
    G = nx.generators.grid_2d_graph(M, N)
    pos = assign_pos(M, N)
    nx.set_node_attributes(G, pos, 'pos')
    
    if not strengthened_region:
        strengthened_region = int(M / 10)
        
    weights = {(u, v): 1. for u, v in G.edges()}
    center = np.floor(M / 2.)
    modified_edges = [edge for edge in G.edges()
                      if ((np.abs(edge[0][0] - center) < strengthened_region) 
                           or (np.abs(edge[1][0] - center) < strengthened_region))]
    
    for e in modified_edges:
        k = np.random.random()
        if k < p:
            weights[e] = w
            
    nx.set_edge_attributes(G, weights, 'weight')
    
    return G


def threshold_model(s_0: np.ndarray, theta: float, G: nx.Graph) -> np.ndarray:
    """Weighted threshold model given by a weighted generalization of the threshold model 
    the one used in DOI: 10.1103/PhysRevLett.113.088701

    Args:
        s_0 (numpy.ndarray): Initial state
        theta (float): Threshold
        G (networkx.Graph): graph

    Returns:
        s_current (numpy.nd.array): Final state 
    """
    
    nodes = list(G.nodes())
    LL = np.array(nx.laplacian_matrix(G).todense(), dtype=float)
    D_inv = np.diag(1/np.diag(LL))
    AA = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
    
    #Transition matrix P
    P = np.matmul(D_inv, AA)
    
    s_current = np.copy(s_0)
    s_old = np.zeros(s_current.shape)
    
    while len(np.nonzero(s_old-s_current)[0]):
        #print(len(np.nonzero(s_old-s)[0]))
        s_old = np.copy(s_current)
        
        for i, node in enumerate(nodes):
            
            neighbors = list(G.neighbors(node))
            neighbor_indices = [nodes.index(n) for n in neighbors]
            
            value = np.sum([P[i, j] * s_old[j]
                            for j in neighbor_indices])
            
            if value > theta:
                s_current[i] = 1.0

    return s_current


def network_evaluate_primal_and_dual_fiedlervalues(primal_graph: nx.Graph, plot_it: bool = False) -> tuple:
    """Get the fielder value for both primal and dual graph.

    Args:
        primal_graph (networkx.Graph): networks representing primal graph
        plot_it (bool, optional): If 'True' plot the resulting dual graph

    Returns:
        primal_fielder_value, dual_fiedler_value: Algebraic connectivity of the primal and dual graph.
    """
    
    dual_graph = create_dual_from_graph(primal_graph)
    
    primal_fiedler_value = nx.algebraic_connectivity(primal_graph)
    dual_fiedler_value = nx.algebraic_connectivity(dual_graph)
    
    return primal_fiedler_value, dual_fiedler_value


def single_threshold_model(NN: int, center: int,
                           strengthened_region: int,
                           prob_disturbed: float, 
                           theta: float, tup_pp_ww: tuple) -> tuple:
    """Run a single threshold model to be used in multiprocessing loop
    """
    
    pp, ww = tup_pp_ww
    
    graph = generate_randomised_lattice(NN, pp, ww)
    
    primal_fiedler_val, dual_fiedler_val = network_evaluate_primal_and_dual_fiedlervalues(graph)
    edges = list(graph.edges())
    
    edge_comm2 = [i for i, edge in enumerate(edges) if ((edge[0][0] - center > strengthened_region + 1) 
                                            or (edge[1][0] - center > strengthened_region + 1))]
    ncomm_2, ncomm2_index = edge_community_to_node_community(graph, edge_comm2)
    
    # Set initial state
    s_0 = np.zeros(len(graph.nodes()))
    indices = np.arange(len(ncomm_2))
    np.random.shuffle(indices)
    
    
    for ii in indices[:int(len(graph.nodes()) * prob_disturbed)]:
        s_0[ncomm2_index[ii]] = 1.
        
    states = threshold_model(s_0, theta, graph)
    
    return tup_pp_ww, primal_fiedler_val, dual_fiedler_val, np.sum(states) / len(graph.nodes())


def run_cascade_multi_single_prob_modi(nprocs: int = 3, NN: int = 10, prob_modi: float = .8, 
                                       iterations: int = 1000, lims_logspace: tuple = (-2, 2),
                                       theta: float = .3, rel_strengthed_region: float = .1,
                                       nn_weight_scan: int=100,
                                       prob_disturbed: float = 1/30., show_progress: bool = True,
                                       save_results: bool = True) -> tuple:
    """Run 

    Args:
        nprocs (int, optional): _description_. Defaults to 3.
        NN (int, optional): _description_. Defaults to 10.
        prob_modi (float, optional): _description_. Defaults to .8.
        iterations (int, optional): _description_. Defaults to 1000.
        lims_logspace (tuple, optional): _description_. Defaults to (-2, 2).
        theta (float, optional): _description_. Defaults to .3.
        rel_strengthed_region (float, optional): _description_. Defaults to .1.
        nn_weight_scan (int, optional): _description_. Defaults to 100.
        prob_disturbed (float, optional): _description_. Defaults to 1/30..
        show_progress (bool, optional): _description_. Defaults to True.
        save_results (bool, optional): _description_. Defaults to True.

    Returns:
        tuple: _description_
    """
    
    MM = 2 * NN + 1
    strengthened_region = NN * rel_strengthed_region
    center = np.floor(MM/2.)
    prob_arr = np.ones(nn_weight_scan) * prob_modi
    weight_arr = np.logspace(min(lims_logspace),
                              max(lims_logspace), 
                              nn_weight_scan)
    
    all_params = np.stack((prob_arr, weight_arr), axis=1)
    
    funci = partial(single_threshold_model, NN, center, strengthened_region, prob_disturbed,
                    theta)
    
    rhos = np.full((iterations, len(weight_arr)),
                    np.nan, dtype=float)
    
    primal_fiedler_arr = np.full((iterations, len(weight_arr)),
                                   np.nan, dtype=float)
    
    dual_fiedler_arr = np.full((iterations, len(weight_arr)),
                                 np.nan, dtype=float)
    
    t0 = time.time()
    with Pool(processes=nprocs) as mpool:
        for kk in tqdm(range(iterations), disable=not show_progress):
            for ii, res_r in enumerate(tqdm(mpool.imap_unordered(funci, all_params),
                                         total=len(all_params), leave=False,
                                         disable=not show_progress)):
                _, ww_r =res_r[0]
                idx_weight = np.argmin(abs(weight_arr - ww_r))

                primal_fiedler_val = res_r[1]
                dual_fiedler_val = res_r[2]
                
                primal_fiedler_arr[kk, idx_weight] = primal_fiedler_val
                dual_fiedler_arr[kk, idx_weight] = dual_fiedler_val 

                rhos[kk, idx_weight] = res_r[3]
            
    if save_results:
        out_path = (out_data_path + "/info_cascade_NN{0}_singlePmodi{1:4f}_".format(NN, prob_modi) + 
                    "theta{0:.2f}_probdist{1:.5f}_iterations{2}.pklz".format(theta, prob_disturbed, iterations))
        with gzip.open(out_path, 'wb') as fh:
            pickle.dump((weight_arr, rhos,
                         primal_fiedler_arr, dual_fiedler_arr), fh)
            
    return weight_arr, prob_arr, primal_fiedler_arr, dual_fiedler_arr, rhos
