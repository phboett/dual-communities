#!/usr/bin/python3
# -*- coding: utf-8 -*

"""Generate the data for the figures with stochastic sources using the model defined
in synthetic leafs."""

import os
import numpy as np

import gzip
import pickle

from tqdm import tqdm

from multiprocessing import Pool as mpool

from dual_communities import synthetic_leafs

from functools import partial
script_path = os
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
                                gamma=.9, threshold=1e-10, link_threshold=1e-8):
    """Optimized edge weights of lattice according to function in synthetic_leafs.py and scan over
    different fluctuation strength to see how the network changes.

    Args:
        sigma_sq_loglims (tuple): Limits in logspace for fluctuation strength.
        nof_sources (int): Number of sources considered. Only supports 2, 3 and 6 due to geometrical constraints.
        nn_vals (int, optional): Number of values of sigma_sq in the scan. Defaults to 50.
        nr_procs (int, optional): Number of processes used by multiprocessing. Defaults to 2.
        KK (float, optional): Scale parameter. Defaults to 500.
        NN (int, optional): Number of nodes. Defaults to 26.
        link_threshold (float): Threshold to remove links that have 
        a negliable capacity after the optimzation algorithm
    """
    
    sigma_sq_arr = np.logspace(min(sigma_sq_loglims),
                               max(sigma_sq_loglims), nn_vals, base=10)
    alpha_funci = lambda sigma_d_sq_r: ((KK**2/(sigma_d_sq_r*nof_sources**3)*(nof_sources- 1 ))
                                        - 1/nof_sources)
    dirichelet_par_arr = np.asarray([alpha_funci(xx) for xx in sigma_sq_arr])
    
    par_arr = np.asarray([xx for xx in zip(sigma_sq_arr, dirichelet_par_arr)])
    
    with mpool(processes=nr_procs) as pool:
        
        res_dict = dict()
        part_func = partial(single_call_syn, NN, KK, nof_sources, gamma)
        
        for ii, res_r in enumerate(tqdm(pool.imap(part_func, par_arr), total=len(par_arr))):
            sigma_sq_r, network_r = res_r
            
            # remove links that fall below threshold
            below_tol_edges = [xx for xx in network_r.edges()
                               if network_r[xx[0]][xx[1]]['weight'] < link_threshold]
            network_r.remove_edges_from(below_tol_edges)
            
            res_dict[sigma_sq_r] = network_r
    
    # Save results
    fpath_out = (script_path + "/data/" +
                 "synthetic_NN{0}_Ns{1}_K{2:.2f}_gamma{3:.4f}_threshold{4:.1E}_linkatol{5:.1E}.pklz".format(NN, nof_sources, KK, gamma,
                                                                                                            threshold, link_threshold))
    with gzip.open(fpath_out, 'wb') as fh_out:
        pickle.dump(res_dict,fh_out)
        
    return


def generate_for_two_sources(nn_vals=50):
    """Generate the data for the system with two stochastic sources."""
    
    sigma_sq_loglims: tuple = (0.5, 5)
    nof_sources = 2
    
    generate_data_multi_sources(sigma_sq_loglims, nof_sources, nn_vals=nn_vals)
    
    return


def generate_for_three_sources(nn_vals=50):
    """Generate the data for the system with three sources."""
    
    sigma_sq_loglims: tuple = (0.5, 5)
    nof_sources = 3
    
    generate_data_multi_sources(sigma_sq_loglims, nof_sources, nn_vals=nn_vals)
    
    return