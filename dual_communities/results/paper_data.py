#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Create data of results presented in publication."""

import numpy as np
import networkx as nx

from scipy.sparse import linalg as sparse_linalg

import gzip
import pickle

from dual_communities.results import leaf_analysis, info_cascade, stochastic_sources

out_data_path = "generated_data"
raw_path = "raw_data"

def sparse_fiedler_vals_n_vecs(laplacian, atol=1e-12, kk=15):
    """Evaluate fielder value and vec of given matrix"""
    
    evals, evecs = sparse_linalg.eigsh(laplacian, sigma=0,  k=kk)
    idx_sort = np.argsort(evals)
    evals = evals[idx_sort]
    evecs = evecs[:, idx_sort]
    
    assert evals[1] > atol
    
    fiedler_val = evals[1]
    fiedler_vec = evecs[:,1]
    
    return fiedler_val, fiedler_vec


def graph_and_dual_eigenvalue_eigenvectors(graph, dual_graph, save_str,
                                           use_sparse=True, save_it=True):
    """Find the fiedlervector and fiedlervalue of primal and dual graph"""
    
    
    sparse_str = ""
    if use_sparse:
        sparse_str = "_sparse"
    
    LL = nx.laplacian_matrix(graph)
    LL_dual = nx.laplacian_matrix(dual_graph)
    
    if use_sparse:
        fiedler_val, fiedler_vec = sparse_fiedler_vals_n_vecs(LL)
        fiedler_val_dual, fiedler_vec_dual = sparse_fiedler_vals_n_vecs(LL_dual)

        
    else:
        fiedler_vec = nx.fiedler_vector(graph)
        fiedler_val = nx.algebraic_connectivity(graph)
        
        fiedler_vec_dual = nx.fiedler_vector(dual_graph)
        fiedler_val_dual = nx.algebraic_connectivity(dual_graph)
        
    if save_it:
        with gzip.open(save_str + "_primal_and_dual_fiedlerval_n_vec{}.pklz".format(sparse_str), "wb") as fh_out:
            pickle.dump((fiedler_val, fiedler_vec, fiedler_val_dual, fiedler_vec_dual), fh_out)
        
    return fiedler_val, fiedler_vec, fiedler_val_dual, fiedler_vec_dual

def all_leaf_primal_n_dual_eigensys():
    
    file_path_list = [('Acer platanoides', 'Maple', 'maple2c_80_graph.pklz'),
                      ('Schizolobium amazonicum', 'Schizolobium amazonicum',
                       'BronxA_125_binary_corrected_graph.pklz'),
                      ('Parkia nitida', 'Parkia nitida a',
                       'BronxA_115_a_binary_corrected_graph.pklz')]
    
    for _, common_name, file_name in file_path_list:
        
        with gzip.open(out_data_path + "/" + file_name, 'rb') as fh_in:
            graph, dual, _, _, _ = pickle.load(fh_in)
            save_str = "{}/{}".format(out_data_path, common_name)
            graph_and_dual_eigenvalue_eigenvectors(graph, dual, save_str,
                                                   use_sparse=True)
            

def generate_all_leaf_data(nr_processes: int = 3):
    
    # Preprocess from raw to graphs
    leaf_analysis.load_and_process_parallel(nr_processes=nr_processes)
    
    # Calculate fiedler values and estimates for topological connectivity
    leaf_analysis.calc_all_fiedler_values_and_estimates()
    
    # Calculate leafs primal n dual fieldervector and fiedlervalue
    all_leaf_primal_n_dual_eigensys()
    
    return


def run_scan_info_cascade(nr_processes: int = 3):
    
    info_cascade.run_cascade_multi_single_prob_modi(nprocs=nr_processes,
                                                    iterations=1000,
                                                    save_results=True)
    
    return

    
def meta_all_data():
    """Generate all data. This may take some time and resources."""
    
    generate_all_leaf_data()
    run_scan_info_cascade()
    stochastic_sources.generate_for_two_sources(iterations=20, nn_vals=50)
    
    return