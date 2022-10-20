#!/usr/bin/python3
# -*- coding: utf-8 -*

"""Generate the data for the figures with stochastic sources using the model defined
in synthetic leafs. The corresponding figure is created creating multiple
runs of different synthetic leafs and averaging the fielder values over the same parameters."""

import os
import numpy as np
import networkx as nx

import gzip
import pickle

from tqdm import tqdm

from multiprocessing import Pool as mpool

from dual_communities import synthetic_leafs
from dual_communities.dual_graph import create_dual_from_graph

from functools import partial

import glob

out_data_path = "generated_data/stochastic_sources"
if not os.path.exists(out_data_path):
    os.mkdir(out_data_path)

def single_call_syn(NN, dirichlet_const, nof_sources, gamma, threshold, pars_tup):
    
    sigma_sq_r, dirich_par = pars_tup
    
    network = synthetic_leafs.build_artificial_leaf_triangular_multisource_dirichlet(NN, gamma=gamma,
                                                                                     nof_sources=nof_sources,
                                                                                     dirichlet_constant=dirichlet_const,
                                                                                     dirichlet_parameter=dirich_par,
                                                                                     threshold=threshold,
                                                                                     mu=-1, sigma=.1)
    
    
    return sigma_sq_r, network


def generate_data_multi_sources(sigma_sq_loglims: tuple, nof_sources: int,
                                nn_vals: int = 50, nr_procs: int = 2, KK: float = 500., NN: int = 26,
                                gamma=.9, threshold=1e-10,  suffix: str = ""):
    """Optimized edge weights of lattice according to function in synthetic_leafs.py and scan over
    different fluctuation strength to see how the network changes.

    Args:
        sigma_sq_loglims (tuple): Limits in logspace for fluctuation strength.
        nof_sources (int): Number of sources considered. Only supports 2, 3 and 6 due to geometrical constraints.
        nn_vals (int, optional): Number of values of sigma_sq in the scan. Defaults to 50.
        nr_procs (int, optional): Number of processes used by multiprocessing. Defaults to 2.
        KK (float, optional): Scale parameter. Defaults to 500.
        NN (int, optional): Number of nodes. Defaults to 26. 
        a negliable capacity after the optimzation algorithm
        suffix (str): str after filename to make multiple runs with same parameters possible
    """
    
    sigma_sq_arr = np.logspace(min(sigma_sq_loglims),
                               max(sigma_sq_loglims), nn_vals, base=10)
    alpha_funci = lambda sigma_d_sq_r: ((KK**2/(sigma_d_sq_r*nof_sources**3)*(nof_sources- 1 ))
                                        - 1/nof_sources)
    dirichelet_par_arr = np.asarray([alpha_funci(xx) for xx in sigma_sq_arr])
    
    par_arr = np.asarray([xx for xx in zip(sigma_sq_arr, dirichelet_par_arr)])

    with mpool(processes=nr_procs) as pool:
        
        res_dict = dict()
        part_func = partial(single_call_syn, NN, KK, nof_sources, gamma, threshold)
        
        for ii, res_r in enumerate(tqdm(pool.imap(part_func, par_arr), total=len(par_arr))):
            sigma_sq_r, network_r = res_r            
            
            res_dict[sigma_sq_r] = network_r
    
    # Save results
    fpath_out = ( out_data_path +
                 "/synthetic_NN{0}_Ns{1}_K{2:.2f}_gamma{3:.4f}_threshold{4:.1E}{5}.pklz".format(NN, nof_sources, KK, gamma,
                                                                                                            threshold,
                                                                                                            suffix))
    with gzip.open(fpath_out, 'wb') as fh_out:
        pickle.dump(res_dict,fh_out)
        
    return


def generate_for_two_sources(nn_vals=50, KK=500, iterations=20, nr_procs=2):
    """Generate the data for the system with two stochastic sources."""
    
    sigma_sq_loglims: tuple = (0.5, 5)
    nof_sources = 2
    for idx in range(iterations):
        generate_data_multi_sources(sigma_sq_loglims, nof_sources, 
                                    nn_vals=nn_vals, suffix="_run{}".format(idx), KK=KK,
                                    nr_procs=nr_procs)
    
    return


def analyse_for_two_sources(NN=26, nof_sources=2, gamma=.9, threshold=1e-10, 
                            link_threshold=1e-8, KK=500):
    """Find the fiedler value for both primal and dual graphs to save them. Links in the 
    primal graph that have a weight below link_threshold will be deleted.

    Args:
        NN (int, optional): _description_. Defaults to 26.
        nof_sources (int, optional): _description_. Defaults to 2.
        gamma (float, optional): _description_. Defaults to .9.
        threshold (_type_, optional): _description_. Defaults to 1e-10.
        link_threshold (_type_, optional): _description_. Defaults to 1e-8.
        KK (int, optional): _description_. Defaults to 500.

    Returns:
        _type_: _description_
    """
    search_pattern = ( out_data_path +
                 "/synthetic_NN{0}_Ns{1}_K{2:.2f}_gamma{3:.4f}_threshold{4:.1E}_linkatol{5:.1E}*.pklz".format(NN, nof_sources, KK, gamma,
                                                                                                            threshold, link_threshold))
    
    file_ls = glob.glob(search_pattern)
    
    total_res_dict = dict()
    
    for file_r in tqdm(file_ls):
        with gzip.open(file_r, 'rb') as fh_in:
            res_dict_r = pickle.load(fh_in)
            
            for key, ele in tqdm(res_dict_r.items(), leave=False):
                gra_r = ele.copy()
                
                fiedler_primal = nx.algebraic_connectivity(gra_r)
                primal_lap_tr = np.trace(nx.laplacian_matrix(gra_r).todense())
                
                below_tol_edges = [xx for xx in gra_r.edges()
                                   if gra_r[xx[0]][xx[1]]['weight'] < link_threshold]
                gra_r.remove_edges_from(below_tol_edges)
                
                dual_graph = create_dual_from_graph(gra_r)
                
                fielder_dual = nx.algebraic_connectivity(dual_graph)
                dual_lap_tr = np.trace(nx.laplacian_matrix(dual_graph).todense())y
                
                fiedler_res_r = (fiedler_primal/primal_lap_tr,
                                 fielder_dual/dual_lap_tr)
                
                key_rounded = np.round(key, decimals=5)
                if key_rounded not in total_res_dict:
                    total_res_dict[key_rounded] = [fiedler_res_r]
                    
                else:
                    total_res_dict[key_rounded].append(fiedler_res_r)
                
                del gra_r
                
    out_fpath = (out_data_path + "/all_synthetic_"+
                 "NN{0}_Ns{1}_K{2:.2f}_gamma{3:.4f}_threshold{4:.1E}_linkatol{5:.1E}.pklz".format(NN, nof_sources, KK, gamma,
                                                                                                            threshold, link_threshold))
     
    with gzip.open(out_fpath, 'wb') as fh_out:
        pickle.dump(total_res_dict, fh_out)
                    
    return total_res_dict