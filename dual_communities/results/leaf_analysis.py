#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Preprocess and analyze the leaf data."""

import numpy as np
import networkx as nx

from scipy.sparse import linalg as sparse_linalg
import os

from multiprocessing import Pool

import gzip
import pickle

import time
from tqdm import tqdm

import pandas as pd

from dual_communities.dual_graph import determine_outer_face, dual_graph_noloops, get_dual_pos
from dual_communities.hierarchy_detection import remove_open_ends, get_hierarchy_levels_dual

from sage.graphs.all import Graph

out_data_path = "generated_data"
raw_path = "raw_data"


def construct_networkx_and_dual_from_leaf_path(path_to_leaf):
    """Read in leaf txt and construct its dual"""

    l1_nodes = pd.read_csv(path_to_leaf+'.node_positions.txt', sep=" ",
                           header=None, names=["id", "posx", "posy"])
    l1_nodes[['id']] = l1_nodes[['id']].astype(np.int32)
    l1_edges = pd.read_csv(path_to_leaf+'.edgelist.txt', sep=" ",
                           header=None, names=["node1", "node2", "thickness", "length"])
    G = nx.Graph()
    for i in range(len(l1_nodes['id'])):
        node = l1_nodes['id'][i]
        G.add_node(node)
        G.nodes[node]['pos'] = [l1_nodes['posx'][i], l1_nodes['posy'][i]]

    G.add_weighted_edges_from(
        list(zip(l1_edges["node1"], l1_edges["node2"], (l1_edges['thickness']/2)**4)))
    ll = {}
    ll.update(zip(list(G.edges()), l1_edges['length']))
    nx.set_edge_attributes(G, ll, 'length')
    
    ##remove small components, such that only one component is left in the end
    print('Number of connected components in G')
    print(len(list(nx.connected_components(G))))
    l = list(nx.connected_components(G))
    print('Removing all components smaller than 100 nodes which are of sizes')
    for component in l:
        if len(component) <= 100:
            G.remove_nodes_from(component)
            
        else:
            print('Big component:')
            print(len(component))
            
    ll = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    ###remove open ends
    kept_indices, G = remove_open_ends(G)
    print(str(len(list(G.nodes()))) +' nodes remaining.')
    G.remove_edges_from(nx.selfloop_edges(G))
    F = Graph(G)
    faces = F.faces()

    outer_face = determine_outer_face(
        faces, pos=nx.get_node_attributes(G, 'pos'))
    print('Constructed faces')
    G_dual_noloops = dual_graph_noloops(
        nx.get_edge_attributes(G, 'weight'), faces, outer_face=outer_face)
    G.graph['faces'] = faces
    print('Constructed dual')

    dual_pos = get_dual_pos(faces, pos=nx.get_node_attributes(G, 'pos'))
    nx.set_node_attributes(G_dual_noloops, name='pos', values=dual_pos)
    for comp in list(nx.connected_components(G_dual_noloops)):
        if len(comp) <= 10:

            G_dual_noloops.remove_nodes_from(comp)
        else:
            print('Big component:')
            print(len(comp))
    print('Done with graph ' + path_to_leaf)
    
    return G, G_dual_noloops


def get_leaf_data_by_path(path, nof_levels=3, verbose=True, save_it=True,
                          file_prefix="", post_matter=True):
    print("## Load Networks")

    start = time.time()
    leaf, leaf_dual = construct_networkx_and_dual_from_leaf_path(path)
    network_construct_time = (time.time() - start)/60.
    if verbose:
        print("Elapsed time {0:.2f}min".format(network_construct_time))

    print("## Hierarchy Detection")
    start = time.time()
    Graph_levels, Dual_levels, boundaries = get_hierarchy_levels_dual(leaf,
                                                                                          leaf_dual,
                                                                                          nof_levels=nof_levels,
                                                                                          use_median=False,
                                                                                          return_fiedler_vecs=False)
    hierarchy_time = (time.time() - start)/60.
    print("Elapsed time {0:.2f}min".format(hierarchy_time))

    if save_it:
        save_path = (out_data_path + "/" +
                           path.split(
                               "/")[-1] + file_prefix + ".pklz")
        print(save_path)

        with gzip.open(save_path, 'wb') as fh:
            pickle.dump([leaf, leaf_dual, [[xx.copy() for xx in yy] for yy in Graph_levels],
                         [[xx.copy() for xx in yy] for yy in Dual_levels], boundaries], fh)


    return leaf, leaf_dual, Graph_levels, Dual_levels, boundaries


def wrapper_multiprocessing(path):

    _ = get_leaf_data_by_path(path, save_it=True, verbose=True)

    return


def load_and_process_parallel(nr_processes=3):
    """use multiprocessing to run in parallel."""


    name_ls = ['maple2c_80_graph',
               'BronxA_078_binary_corrected_graph', 'BronxA_115_a_binary_corrected_graph',
               'BronxA_125_binary_corrected_graph']
    
    file_path_ls = [raw_path + "/leafs/" + xx for xx in name_ls]

    with Pool(processes=nr_processes) as pool:
        for res_idx in pool.imap(wrapper_multiprocessing, file_path_ls):
            pass

    return

def calc_dual_fiedler_estimate(nodes_in_comm1, nodes_in_comm2, primal_boundary_edges, primal_weight_dict, clean_boundary_duplicates=True):
    """
    Evaluate estimate of dual fiedler value as topological connectivity.
    """

    # Locate duplicates in the boundary 
    if clean_boundary_duplicates:
        boundary_set = set([tuple(sorted(xx)) for xx in primal_boundary_edges])
        primal_boundary_edges = list(boundary_set)

    inverse_boundary_edge_weight = np.zeros(len(primal_boundary_edges))
    for count, edge in enumerate(primal_boundary_edges):
        try:
            inverse_boundary_edge_weight[count] = 1/primal_weight_dict[edge]
        except KeyError:
            inverse_boundary_edge_weight[count] = 1 / \
                primal_weight_dict[edge[::-1]]
                
    return (nodes_in_comm1 + nodes_in_comm2) / (nodes_in_comm1 * nodes_in_comm2) * np.sum(inverse_boundary_edge_weight)


def calculate_leaf_fielder_val_and_estimate(leaf, leaf_dual, dual_levels, boundaries, verbose=False,
                                            use_sparse=False):
    weights = nx.get_edge_attributes(leaf, 'weight')
    leaf_dual_Fiedler_estimate = calc_dual_fiedler_estimate(
        len(dual_levels[1][0].nodes()),
        len(dual_levels[1][1].nodes()),
        boundaries[0][0],
        weights)

    if use_sparse:
        L_dual = nx.laplacian_matrix(leaf_dual)
        ws, _ = sparse_linalg.eigsh(L_dual, sigma=0, k=15)

    else:
        L_dual = nx.laplacian_matrix(leaf_dual).A
        ws = np.linalg.eigvalsh(L_dual)

    leaf_dual_eigval_spec = ws

    return leaf_dual_eigval_spec, leaf_dual_Fiedler_estimate


def calc_all_fiedler_values_and_estimates(file_path_list=None, use_sparse=True):
    """Calculate the fiedler value (and entire Laplacian spectrum) and the estimate for the leaf examples."""

    file_path_list = [('Schizolobium amazonicum', 'BronxA_125_binary_corrected_graph_fiedlerestimate_figure.pklz'),
                      ('Parkia nitida a', 'BronxA_115_a_binary_corrected_graph_fiedlerestimate_figure.pklz'),
                      ('Acer platanoides', 'maple2c_80_graph_fiedlerestimate_figure.pklz')]

    for name, file_r in tqdm(file_path_list):
        with gzip.open(os.path.join(out_data_path, file_r), "rb") as fh:
            data = pickle.load(fh)

        lambda_2, estimate_lambda_2 = calculate_leaf_fielder_val_and_estimate(
            data[0], data[1], data[3], data[4], use_sparse=use_sparse)

        sparse_str = ""
        if use_sparse:
            sparse_str = "_sparse"


        with gzip.open(out_data_path + "/{0}_fiedler_and_estimate{1}.pklz".format(name, sparse_str), "wb") as fh_out:
            pickle.dump((name, lambda_2, estimate_lambda_2), fh_out)

    return
