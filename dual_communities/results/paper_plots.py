#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Script that produces the plots in the manuscript."""

import os
import copy
import gc

import numpy as np
import networkx as nx

import gzip
import pickle

import matplotlib as mpl
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib import colors as mplcolors

from dual_communities.plot_functions import draw_actual_leaf
from dual_communities.hierarchy_detection import get_hierarchy_levels_dual
from dual_communities.dual_graph import get_boundary_infos, create_dual_from_graph
from dual_communities.distribution_factors import calc_LODF_matrix, redefined_index
from dual_communities.results import info_cascade as _inf_model
from dual_communities.results.leaf_analysis import calculate_leaf_fielder_val_and_estimate, calc_dual_fiedler_estimate
from dual_communities.tools import assign_pos

from tqdm import tqdm

out_data_path = "dual_communities/results/data"
raw_path = "dual_communities/results/raw_data" 

if not os.path.exists("plots/"):
    os.mkdir("plots/")

fiedler_primal_cmap = plt.get_cmap("coolwarm")
fiedler_dual_cmap = plt.get_cmap("PiYG")
cut_cmap = plt.get_cmap('plasma_r')
cmap_under = 'darkgray'

mpl.style.use('default')
plt.rc('text', usetex = True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')

out_data_path = "generated_data"
raw_path = "raw_data"

if not os.path.exists("plots/"):
    os.mkdir("plots/")


class MidpointNormalize(mplcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mplcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    

def connection_communities_network_robustness(calculate_lodfs: bool = False,
                                              save_fig: bool = False, vlims: tuple=(1e-8, 1)):
    """Demonstrate the connection of communities and network robustness."""
    
    color_ls = ["#1b9e77", "#d95f02", "#7570b3"]
    color_ls_net = ["#8dd3c7", "#ffffb3", "#bebada"]
    
    # Load and calculate LODF for both networks
    leaf_path_name = out_data_path + "/BronxA_125_binary_corrected_graph.pklz"
    failing_link_leaf = (15118, 8087)
    failing_link_scandinavia = ('6592', '6861')
    
    
    with gzip.open(leaf_path_name,
              'rb') as fh_leaf:
        leaf_graph, _, _, _, _ = pickle.load(fh_leaf)
        
    
    with open(raw_path + "/Scandinavia_lineloading_80percent.p",
                   'rb') as fh_scandi:
        scandinavia_graph = pickle.load(fh_scandi)
    
    lodf_path = out_data_path + "/fig1_lodf_matrices_amazonicum_scandinavia.pklz"
    if calculate_lodfs:
        leaf_LODF = calc_LODF_matrix(leaf_graph)
        scandi_LODF = calc_LODF_matrix(scandinavia_graph)
        
        with gzip.open(lodf_path, 'wb') as fh_out:
            pickle.dump((leaf_LODF, scandi_LODF), fh_out)
            
    else:
        with gzip.open(lodf_path, 'rb') as fh_in:
            leaf_LODF, scandi_LODF = pickle.load(fh_in)
        pass
        
    
    leaf_link = redefined_index(list(leaf_graph.edges()),
                                       failing_link_leaf)
    scandi_link = redefined_index(list(scandinavia_graph.edges()), 
                                        failing_link_scandinavia)
    
    # Plot it
    fig = plt.figure(figsize=(20, 8))
    
    gs = fig.add_gridspec(1, 3, width_ratios=[1., 1., 1.], wspace=.3)
    gs_lodf = gs[0].subgridspec(2, 2)
    gs_cascade = gs[1].subgridspec(3, 2)
    gs_cascade_avg = gs[2].subgridspec(3, 1)
    
    ax_top_scandi = fig.add_subplot(gs_lodf[0,0])
    ax_top_maple = fig.add_subplot(gs_lodf[1,0])
    
    ax_lodf_scandi = fig.add_subplot(gs_lodf[0,1])
    ax_lodf_maple = fig.add_subplot(gs_lodf[1,1])
    
    ax_casc_network_wlow_start = fig.add_subplot(gs_cascade[0, 0])
    ax_casc_network_wmid_start = fig.add_subplot(gs_cascade[1, 0])
    ax_casc_network_whigh_start = fig.add_subplot(gs_cascade[2, 0])
    
    ax_casc_network_wlow_end = fig.add_subplot(gs_cascade[0, 1])
    ax_casc_network_wmid_end = fig.add_subplot(gs_cascade[1, 1])
    ax_casc_network_whigh_end = fig.add_subplot(gs_cascade[2, 1])
    
    ax_casc_average = fig.add_subplot(gs_cascade_avg[:, 0])
    ax_casc_avg_pos = ax_casc_average.get_position()
    
    ax_casc_avg_pos_new = [ax_casc_avg_pos.x0 + 0.03, ax_casc_avg_pos.y0,
                           ax_casc_avg_pos.width + .02, ax_casc_avg_pos.height]
    ax_casc_average.set_position(ax_casc_avg_pos_new)
    
    
    ## Networks
    pos_scandi = nx.get_node_attributes(scandinavia_graph, 'pos')
    cap_scandi = np.asarray([scandinavia_graph[uu][vv]['weight'] 
                           for uu, vv in scandinavia_graph.edges()])
    nx.draw_networkx_edges(scandinavia_graph, pos=pos_scandi, ax=ax_top_scandi,
                           width=np.sqrt(cap_scandi/cap_scandi.max()) * 5)
    
    cap_leaf = np.asarray([leaf_graph[uu][vv]['weight']
                            for uu, vv in leaf_graph.edges()])
    pos_leaf = nx.get_node_attributes(leaf_graph, 'pos')
    nx.draw_networkx_edges(leaf_graph, pos=pos_leaf, ax=ax_top_maple,
                           width=np.sqrt(cap_leaf/cap_leaf.max()) * 5)
    
    ## LODFS
    cmap_lodf = copy.copy(plt.get_cmap('cividis'))
    cmap_lodf.set_bad('r', 1.0)
    cmap_lodf.set_over('r', 1.0)
    cmap_lodf.set_under('gainsboro',1.0)
    
    color_scandi = np.array([np.log10(x) 
                  if x > 1e-8 else -np.inf 
                  for x in np.abs(scandi_LODF[:, scandi_link])])
    color_scandi[scandi_link] = 1e5
    scandi_width = [1.5] * len(scandinavia_graph.edges())
    scandi_width[scandi_link] = 3.
     
    vmax_scandi = 10**(np.max(color_scandi[np.logical_and(np.logical_and(~np.isnan(color_scandi),
                                                                  color_scandi!=-float("inf")),
                                                   color_scandi!=1e5)]))
    vmin_scandi = 10**(np.min(color_scandi[np.logical_and(~np.isnan(color_scandi),
                                                   color_scandi!=-float("inf"))]))
    
    color_leaf = np.array([np.log10(x) 
                  if x > 1e-8 else -np.inf 
                  for x in np.abs(leaf_LODF[:, leaf_link])])
    color_leaf[leaf_link] = 1e5
    leaf_width = [1.5] * len(leaf_graph.edges())
    leaf_width[leaf_link] = 5.
    vmax_leaf = 10**(np.max(color_leaf[np.logical_and(np.logical_and(~np.isnan(color_leaf),
                                                                  color_leaf!=-float("inf")),
                                                   color_leaf!=1e5)]))
    vmin_leaf = 10**(np.min(color_leaf[np.logical_and(~np.isnan(color_leaf),
                                                   color_leaf!=-float("inf"))]))
    
    if vlims is None:
        vmin= min(vmin_leaf, vmin_scandi)
        vmax = max(vmax_leaf, vmax_scandi)
        
    else:
        vmin = min(vlims)
        vmax = max(vlims)
    
    nx.draw_networkx_edges(scandinavia_graph, pos_scandi, ax=ax_lodf_scandi,
                           edge_color=color_scandi,
                           edge_cmap=cmap_lodf, edge_vmin=np.log10(vmin), edge_vmax=np.log10(vmax),
                           width=scandi_width)
    
    
    edges_leaf_lodf = nx.draw_networkx_edges(leaf_graph, pos_leaf, ax=ax_lodf_maple,
                           edge_color=color_leaf, edge_cmap=cmap_lodf,
                           edge_vmin=np.log10(vmin), edge_vmax=np.log10(vmax),
                           width=leaf_width)
    
    for ax_r in [ax_top_maple, ax_top_scandi,
                 ax_lodf_maple, ax_lodf_scandi]:
        ax_r.axis('off')
        
    ax_top_maple.set_rasterized(True)
    ax_lodf_maple.set_rasterized(True)        
    
    # Cascade results
    
    ## Initial networks
    seed = 114
    weight_ls = [0.01, 1, 100]
    ax_net_ls = [ax_casc_network_wlow_start, ax_casc_network_wmid_start,
                 ax_casc_network_whigh_start]
    ax_net_end_ls = [ax_casc_network_wlow_end, ax_casc_network_wmid_end,
                 ax_casc_network_whigh_end]
    width_norm = lambda val_r: (np.log10(val_r) + 2)/1. + .75
    NN = 4
    for uu, weight_r in enumerate(weight_ls):
        MM = 2 * NN + 1
        str_region = 1
        center = np.floor(MM / 2.)
        graph_r = _inf_model.generate_randomised_lattice(NN, .8, 
                                                               weight_r, strengthened_region=str_region,
                                                               seed=seed)
        width_ls = np.array([width_norm(graph_r[uu][vv]['weight']) 
                             for uu, vv in graph_r.edges()])
        
        ax_net_ls[uu].text(-.2, .5, "$w_\\ell = 10^{" + "{0:d}".format(int(np.log10((weight_ls[uu]))))
                                 + "}$", transform=ax_net_ls[uu].transAxes, rotation=90,
                                 verticalalignment='center', fontsize=30)
        
        # Set intial state
        MM = 2 * NN + 1
        str_region = 2
        center = np.floor(MM / 2.)
        
        edges = graph_r.edges()
        edge_comm2 = [ii for ii, edge_r in enumerate(edges) if ((edge_r[0][0] - center > str_region + 1)
                                                                or edge_r[1][0] - center > str_region + 1)]
        ncomm_2, ncomm2_index = _inf_model.edge_community_to_node_community(graph_r,
                                                                            edge_comm2)
        
        s_0 = np.zeros(len(graph_r.nodes()))
        indices = np.arange(len(ncomm_2))
        
        np.random.shuffle(indices)
        for ii in indices[:int(len(graph_r.nodes()) * .05)]:
            s_0[ncomm2_index[ii]] = 1.
        
        
        state = _inf_model.threshold_model(s_0, .3, graph_r)
        
        pos_grid = nx.get_node_attributes(graph_r, 'pos')
        node_color_start = np.asarray([color_ls[0]] * len(graph_r.nodes()))
        node_color_start[np.where(abs(s_0 - 1) < 1e-8)] = color_ls[1]
        
        node_shape_start_active = [xx for kk, xx in enumerate(graph_r.nodes())
                                   if abs(s_0[kk] - 1) < 1e-8]
        node_shape_start_nonactive = [xx for kk, xx in enumerate(graph_r.nodes())
                                   if abs(s_0[kk]) < 1e-8]
        nx.draw_networkx_edges(graph_r, pos_grid,
                               ax=ax_net_ls[uu], width=width_ls)
        nx.draw_networkx_nodes(graph_r, pos=pos_grid,
                               ax=ax_net_ls[uu], node_size=100,
                               node_color=color_ls_net[1],
                               nodelist=node_shape_start_active,
                               node_shape='v',
                               linewidths=1., edgecolors='k')
        nx.draw_networkx_nodes(graph_r, pos=pos_grid,
                               ax=ax_net_ls[uu], node_size=80,
                               node_color=color_ls_net[0], nodelist=node_shape_start_nonactive,
                               node_shape='o',
                               linewidths=1., edgecolors='k')
        
        node_color_end = np.asarray([color_ls_net[0]] * len(graph_r.nodes()))
        node_color_end[np.where(abs(state - 1) < 1e-8)] = color_ls_net[1] 
        
        node_shape_end_active = [xx for kk, xx in enumerate(graph_r.nodes())
                                   if abs(state[kk] - 1) < 1e-8]
        node_shape_end_nonactive = [xx for kk, xx in enumerate(graph_r.nodes())
                                   if abs(state[kk]) < 1e-8]
        
        
        nx.draw_networkx_edges(graph_r, pos=pos_grid, ax=ax_net_end_ls[uu],
                               width=width_ls)
        nx.draw_networkx_nodes(graph_r, pos=pos_grid, ax=ax_net_end_ls[uu],
                               node_size=100, node_color=color_ls_net[1],
                               nodelist=node_shape_end_active, node_shape='v',
                               linewidths=1., edgecolors='k')
        nx.draw_networkx_nodes(graph_r, pos=pos_grid, ax=ax_net_end_ls[uu],
                               node_size=80, node_color=color_ls_net[0],
                               nodelist=node_shape_end_nonactive, node_shape='o',
                               linewidths=1., edgecolors='k')
        
        
        ax_net_ls[uu].invert_xaxis()
        ax_net_end_ls[uu].invert_xaxis()
        ax_net_ls[uu].axis('off')
        ax_net_end_ls[uu].axis('off')
    
    ax_net_ls[-1].text(.5, -.3, "$t_0$", transform=ax_net_ls[uu].transAxes, rotation=0,
                                 horizontalalignment='center', fontsize=30)
    
    ax_net_end_ls[-1].text(.5, -.3, "$t_\\infty$", transform=ax_net_end_ls[uu].transAxes, rotation=0,
                                 horizontalalignment='center', fontsize=30)
    
    ## Average rho_inf plot
    with gzip.open(out_data_path + 
                   "/info_cascade_NN10_singlePmodi0.8000_theta0.30" + 
                   "_probdist0.05000.pklz", 'rb') as fh_avg:
        weight_arr, rhos, _, _ = pickle.load(fh_avg)
    
    ax_casc_average.plot(np.quantile(rhos, q=.5, axis=0), weight_arr, lw=2, c=color_ls[2])
    ax_casc_average.fill_betweenx(weight_arr, np.quantile(rhos, q=0.25, axis=0),
                                  np.quantile(rhos, q=.75, axis=0), color=color_ls[2],
                                  alpha=.5)
    
    ax_casc_average.invert_yaxis()
    
    ax_casc_average.set_yscale('log')
    
    ax_casc_average.tick_params(axis='both', which='major', labelsize=26)
    ax_casc_average.tick_params(axis='both', which='minor', labelsize=26)
    
    ax_casc_average.set_ylabel("$w_\\ell$", size=30)
    ax_casc_average.set_xlabel("$\\rho_\\infty$", size=30)
    
    ax_casc_average.axhline(y=0, linestyle='--', color='k', lw=2.)
    
    #fig.tight_layout()
    
    pos_ax_lodf = ax_lodf_maple.get_position()
    shrink_cb = .03
    ax_flow_cbar = fig.add_axes([pos_ax_lodf.x0 + shrink_cb/2., pos_ax_lodf.y0 - .05,
                                 pos_ax_lodf.x1 - pos_ax_lodf.x0 - shrink_cb, .01])
    fig.text(pos_ax_lodf.x0 + shrink_cb/2. - .06, pos_ax_lodf.y0 - 0.05, r'$|\Delta F|$', 
             size=30, weight='bold', verticalalignment='center')
    edges_leaf_lodf.cmap = cmap_lodf
    edges_leaf_lodf.set_clim(vmin, vmax)
    edges_leaf_lodf.norm = mplcolors.LogNorm(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(edges_leaf_lodf, cax=ax_flow_cbar,
                      orientation='horizontal')
    cb.ax.tick_params(labelsize=20, width=1.0)
    cb.ax.axis('on')
    
    cb.ax.set_xticks([10**-8, 10**-4, 1])

    
    # Aesthetics
    panel_label_ls = ["$\\textbf{a}$", "$\\textbf{c}$", "$\\textbf{b}$",
                      "$\\textbf{d}$", "$\\textbf{e}$", "$\\textbf{f}$",
                      "$\\textbf{g}$", "$\\textbf{h}$", "$\\textbf{i}$"]
    
    all_axs_ls = [ax_top_scandi, ax_lodf_scandi]
    panel_idx = 0
    for uu, ax_r in enumerate(all_axs_ls):
        posi = ax_r.get_position()
        fig.text(posi.x0, .925, panel_label_ls[uu],
                 fontsize=30, weight='bold')
        panel_idx += 1
        
    for uu, ax_r in enumerate([ax_top_maple, ax_lodf_maple]):
        posi = ax_r.get_position()
        fig.text(posi.x0, .45, panel_label_ls[panel_idx],
                 fontsize=30, weight='bold')
        panel_idx += 1
        
    casc_ax_ls = [ax_casc_network_wlow_start,
                  ax_casc_average]
    for uu, ax_r in enumerate(casc_ax_ls):
        posi = ax_r.get_position()
        fig.text(posi.x0 - .025, .925, panel_label_ls[panel_idx],
                 fontsize=30, weight='bold')
        panel_idx +=1
        
        
    if save_fig:
        fig_path = "plots/primal_dual_cascades.pdf"
        fig.savefig(fig_path, transparent=True, bbox_inches='tight')
        
        fig.clear()
        plt.close(fig)
        
    else:
        plt.show()
    
    return


def primal_and_dual_communities_in_spatial_networks(save_fig: bool = False):
    """Figure highlighting primal and dual communities in spatial
    networkx in an tirivial example, a leaf and the EU power grid."""
    
    # Load Europe data
    with open(raw_path + '/European_grid/' + 
              'Europe_pypsa_boundary_colors.pickle', 'rb') as file:
        boundary_dictionary_Europe = pickle.load(file)
        
    with open(raw_path + '/European_grid/' + 
              'Europe_pypsa_boundary_width.pickle', 'rb') as file:
        boundary_width_Europe = pickle.load(file)
    
    with open(raw_path + '/European_grid/' + 
              'Europe_pypsa_dual_boundary_colors.pickle', 'rb') as file:
        boundary_dictionary_Europe_dual = pickle.load(file)
        
    with open(raw_path + '/European_grid/' + 
              'Europe_pypsa_dual_boundary_width.pickle', 'rb') as file:
        boundary_width_Europe_dual = pickle.load(file)    

    with open(raw_path + '/European_grid/' + 
              'Europe_pypsa_all_duals.pickle','rb') as fh:
        Europe_all_duals = pickle.load(fh)
    
    with open(raw_path + '/European_grid/' + 
              'Europe_pypsa_all_primals.pickle','rb') as fh:
        Europe_all_primals = pickle.load(fh)
        
        
    # Plotting
    fig, ax = plt.subplots(3,6,figsize=(24,12))
    
    # # Panel a)
    with open(raw_path + "/Small_sample_network_mod.pickle", 'rb') as fh_small:
        graph_small = pickle.load(fh_small)
        
    with open(raw_path + "/Small_sample_network_mod_dual.pickle", 'rb') as fh_small_dual:
        graph_small_dual = pickle.load(fh_small_dual)
        
    cmap = plt.get_cmap("Greens")
    color1 = cmap(.3)
    color2 = cmap(.9)
    
    pos = nx.get_node_attributes(graph_small, 'pos')
    
    line_capacities = np.array([graph_small[uu][vv]['weight'] for uu, vv in graph_small.edges()])
    line_capacities[line_capacities < 1] = 1
    
    edge_vmax = np.max(line_capacities)
    
    nx.draw_networkx_nodes(graph_small, pos,
                           ax=ax[0,0],
                           node_size=100,
                           node_color='grey')
    nx.draw_networkx_edges(graph_small, pos, ax=ax[0,0],
                           edge_color='black',
                           width=np.sqrt(line_capacities/edge_vmax)*5)
    
    alpha_strength = 0.5
    es = [(2,1),(1,3)]
    es_weak = [(10, 2), (6, 1), (5, 8)]    
    weak_graph = nx.Graph()
    weak_graph.add_edges_from(es_weak)
    nx.draw_networkx_nodes(weak_graph, 
                           pos,
                           ax = ax[0,0], 
                           node_size = 0)

    nx.draw_networkx_edges(weak_graph,
                                pos,
                                ax = ax[0,0],
                                width = 14.,
                                alpha = alpha_strength,
                                edge_color = [color1]*4)
    
    strong_graph = nx.Graph()
    strong_graph.add_edges_from(es)
    nx.draw_networkx_nodes(strong_graph, 
                                pos,
                                ax = ax[0,0], 
                                node_size = 0)

    nx.draw_networkx_edges(strong_graph,
                                pos,
                                ax = ax[0,0],
                                width = 14.,
                                alpha = alpha_strength,
                                edge_color = [color2]*2)
    
    ax[0,0].plot([],[], color = color1, label = 'weak links', lw = 10, alpha = 0.8)
    ax[0,0].plot([],[], color = color2, label = 'strong links', lw = 10, alpha = 0.8)

    ax[0,0].legend(fontsize = 20,fancybox = True, loc = 'lower right',
                   bbox_to_anchor = (0.95, 0.8, 0.5, 0.5))

    ##############################################################

    dual_pos = nx.get_node_attributes(graph_small_dual, 'pos')
    line_capacities = np.array([graph_small_dual[u][v]['weight'] for u,v in graph_small_dual.edges()])
    es_weak_dual = [list(graph_small_dual.edges())[i] for i in np.where(np.array(line_capacities) == 0.2)[0]]
    es_strong_dual = [list(graph_small_dual.edges())[i] for i in np.where(np.array(line_capacities) == 5.)[0]]

    line_capacities[line_capacities<1] = 1  


    edge_vmax = np.max(line_capacities)
    nx.draw_networkx_nodes(graph_small_dual, 
                                dual_pos,
                                ax = ax[0,1], 
                                node_size = 100,
                                node_shape = 's',
                                node_color = 'grey')

    nx.draw_networkx_edges(graph_small_dual,
                                   dual_pos,
                                   ax = ax[0,1],
                                   #width = 2,
                                   width = np.sqrt(line_capacities/edge_vmax)*5)

    nx.draw_networkx_nodes(graph_small, 
                                pos,
                                ax = ax[0,1], 
                                node_size = 0)
    nx.draw_networkx_edges(graph_small,
                                pos,
                                ax = ax[0,1],
                                width = 2.0,
                                #style = 'dotted',
                                alpha = 0.2)

    weak_graph = nx.Graph()
    weak_graph.add_edges_from(es_weak_dual)
    nx.draw_networkx_nodes(weak_graph, 
                           dual_pos,
                           ax = ax[0,1], 
                           node_size = 0)

    nx.draw_networkx_edges(weak_graph,
                                dual_pos,
                                ax = ax[0,1],
                                width = 14.,
                                alpha = alpha_strength,
                                edge_color = [color1]*len(weak_graph.edges()))
    strong_graph = nx.Graph()
    strong_graph.add_edges_from(es_strong_dual)
    
    nx.draw_networkx_nodes(strong_graph, 
                                dual_pos,
                                ax = ax[0,1], 
                                node_size = 0)

    nx.draw_networkx_edges(strong_graph,
                            dual_pos,
                            ax = ax[0,1],
                            width = 14.,
                            alpha = alpha_strength,
                            edge_color = [color2]*len(strong_graph.edges()))

    ###################################### d)
    v = nx.fiedler_vector(graph_small)
    vmin = np.min(v)
    vmax = np.max(v)
    norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)   


    nx.draw_networkx_nodes(graph_small, 
                                   pos,
                                   ax = ax[1,0], 
                                   node_size = 300,
                                   node_color = norm(v),
                                   cmap = fiedler_primal_cmap)
    nx.draw_networkx_edges(graph_small,
                                   pos,
                                   ax = ax[1,0],
                                   width = 1)
    
    ###################################### 
    v = nx.fiedler_vector(graph_small_dual)
    vmin = np.min(v)
    vmax = np.max(v)
    norm = MidpointNormalize(midpoint = 0, vmin = vmin, vmax = vmax)   
    nx.draw_networkx_nodes(graph_small, 
                           pos,
                           ax = ax[1,1], 
                           node_size = 0)

    nx.draw_networkx_edges(graph_small,
                           pos,
                           ax = ax[1,1],
                           width = 2.0,
                           alpha = 0.3)

    nx.draw_networkx_nodes(graph_small_dual, 
                           dual_pos,
                           ax = ax[1,1], 
                           node_size = 200,
                           node_color = norm(v),
                           cmap = fiedler_dual_cmap,
                           node_shape = 's')
    
    nx.draw_networkx_edges(graph_small_dual,
                           dual_pos,
                           ax = ax[1,1],
                           width = 1)

    ###################################################
    
    nx.draw_networkx_nodes(graph_small, 
                           pos,
                           ax = ax[2,0], 
                           node_size = 100,
                           node_color = 'grey')
    
    nx.draw_networkx_edges(graph_small,
                           pos,
                           ax = ax[2,0],
                           width = 1)

    nx.draw_networkx_nodes(graph_small, 
                                   pos,
                                   ax = ax[2,1], 
                                   node_size = 100,
                                   node_color = 'grey')

    nx.draw_networkx_edges(graph_small,
                                   pos,
                                   ax = ax[2,1],
                                   width = 1)

    del graph_small, graph_small_dual
    
    ## Leaf
    # Load maple 
    with gzip.open(out_data_path + "/maple2c_80_graph.pklz", "rb") as fh_maple:
        maple_gra, maple_gra_dual, maple_gra_lvls, maple_gra_dual_lvls, maple_boundaries = pickle.load(fh_maple)
    ## Fiedler
    with gzip.open(out_data_path + "/Maple_primal_and_dual_fiedlerval_n_vec_sparse.pklz", "rb") as fh_maple_fiedler:
        [_, maple_fiedler_vec,
         _, maple_fiedler_vec_dual] = pickle.load(fh_maple_fiedler)
        
    with gzip.open(out_data_path + "/Maple_primal_clustering_fiedler.pklz", "rb") as fh_maple_pc:
        [_, maple_prim_boundaries, _] = pickle.load(fh_maple_pc)
        
    # b)
    pos = nx.get_node_attributes(maple_gra,'pos')

    capacities = [maple_gra[u][v]['weight'] for u,v in maple_gra.edges()]   
    vmax = np.max(capacities)

    nx.draw_networkx_nodes(maple_gra, 
                               pos = pos,
                               ax = ax[0,2],
                               node_size = 0, 
                               cmap=fiedler_primal_cmap)

    nx.draw_networkx_edges(maple_gra, 
                                   pos=pos,
                                   ax=ax[0,2], 
                                   width=np.sqrt(capacities/vmax)*15,
                                   edge_color='black')
    
    dual_pos = nx.get_node_attributes(maple_gra_dual,'pos')

    capacities = [maple_gra_dual[u][v]['weight'] for u,v in maple_gra_dual.edges()]   
    vmax = np.max(capacities)
    nx.draw_networkx_nodes(maple_gra_dual, 
                               pos = dual_pos,
                               ax = ax[0,3],
                               node_size = 0, 
                               cmap = fiedler_primal_cmap)

    nx.draw_networkx_edges(maple_gra_dual, 
                               pos = dual_pos,
                               ax = ax[0,3], 
                               edge_color = 'black',
                               width = np.sqrt(capacities/vmax)*1)


    # e) Leaf: Fiedlervector primal and dual graph
    
    minval = np.min(maple_fiedler_vec)
    maxval = np.max(maple_fiedler_vec)
    norm = MidpointNormalize(midpoint = 0, vmin = minval, vmax = maxval)    

    nx.draw_networkx_nodes(maple_gra, 
                                pos = pos,
                                ax = ax[1,2],
                                node_color = norm(maple_fiedler_vec),
                                node_size = 20, 
                                cmap = fiedler_primal_cmap)

    nx.draw_networkx_edges(maple_gra, 
                                pos = pos,
                                ax = ax[1,2], 
                                edge_color = 'black')

    minval = np.min(maple_fiedler_vec_dual)
    maxval = np.max(maple_fiedler_vec_dual)
    norm = MidpointNormalize(midpoint = 0, vmin = minval, vmax = maxval)    

    nx.draw_networkx_nodes(maple_gra_dual, 
                                pos = dual_pos,
                                ax = ax[1,3],
                                node_color = norm(maple_fiedler_vec_dual),
                                node_size = 20, 
                                cmap = fiedler_dual_cmap)

    nx.draw_networkx_edges(maple_gra_dual, 
                                   pos = dual_pos,
                                   ax = ax[1,3], 
                                   edge_color = 'black')
    
    ##  Partitions
    ## Primal side
    partition_levels=3
    maple_prim_boundary_dict, maple_prim_boundary_width = get_boundary_infos(maple_gra, maple_prim_boundaries,
                                                                            cut_lvl=partition_levels)
    
    width_maple_prim = np.array([maple_prim_boundary_width[(u, v)]*2 
                                 for u, v in maple_gra.edges()])
    
    width_maple_prim[width_maple_prim==0] = 1.
    width_maple_prim[width_maple_prim==5] = 10.
    pos_maple = nx.get_node_attributes(maple_gra,'pos')
    
    nof_levels = partition_levels
    cmap = cut_cmap
    cm = mpl.colors.ListedColormap([cmap(i/nof_levels) for i in range(nof_levels)])
    cm.set_under(cmap_under,1.0)
    edge_vmin = 0
    edge_vmax = partition_levels

    nx.draw_networkx_edges(maple_gra, pos,
                           ax=ax[2, 2],
                           width=1., edge_color=cmap_under)
    
    boundary_edges = [key for key, xx in maple_prim_boundary_dict.items()
                      if xx >= 0]
    sub_graph = maple_gra.edge_subgraph(boundary_edges)
    for (uu, vv), ele in maple_prim_boundary_dict.items():
        if ele >= 0:
            sub_graph[uu][vv]['edge_color'] = ele

    color_sub = [sub_graph[uu][vv]['edge_color'] for uu, vv in sub_graph.edges()]
    
    nx.draw_networkx_edges(sub_graph, 
                           nx.get_node_attributes(sub_graph, 'pos'),
                           ax=ax[2, 2],
                           width=3.,
                           edge_color=color_sub,
                           edge_cmap=cm,
                           edge_vmin=edge_vmin,
                           edge_vmax=edge_vmax)

    # # Dual side
    maple_boundary_dict, maple_boundary_width = get_boundary_infos(maple_gra, maple_boundaries,
                                                                   cut_lvl=partition_levels)
    width = np.array([maple_boundary_width[(u, v)]*4 for u, v in maple_gra.edges()])
    
    width[width==0] = 1.
    width[width==5] = 10.
    pos_maple = nx.get_node_attributes(maple_gra,'pos')

    nof_levels = partition_levels
    cmap = cut_cmap
    cm = mpl.colors.ListedColormap([cmap(i/nof_levels) for i in range(nof_levels)])
    cm.set_under(cmap_under, 1.0)
    edge_vmin = 0
    edge_vmax = partition_levels
    
    nx.draw_networkx_edges(maple_gra, pos_maple, ax=ax[2, 3],
                           width=1., edge_color=cmap_under)
    
    boundary_edges = [key for key, xx in maple_boundary_dict.items()
                      if xx >=0]
    sub_graph = maple_gra.edge_subgraph(boundary_edges)
    for (uu, vv), ele in maple_boundary_dict.items():
        if ele >= 0:
            sub_graph[uu][vv]['edge_color'] = ele

    color_sub = [sub_graph[uu][vv]['edge_color'] for uu, vv in sub_graph.edges()]
    
    nx.draw_networkx_edges(sub_graph, 
                           pos=nx.get_node_attributes(sub_graph, 'pos'),
                           ax = ax[2, 3],
                           width=3.,
                           edge_color=color_sub,
                           edge_cmap=cm,
                           edge_vmin=edge_vmin,
                           edge_vmax=edge_vmax)

    
    
    del maple_gra, maple_gra_dual, maple_gra_lvls, maple_gra_dual_lvls, maple_boundaries
    gc.collect()
    
    # Europe Example (c,f,i)
    B = Europe_all_primals[0][0]
    pos = nx.get_node_attributes(B,'pos')

    line_capacities = [B[u][v]['weight'] for u,v in B.edges()]
    vmax = np.max(line_capacities)

    nx.draw(B,
        ax = ax[0,4],
        pos = nx.get_node_attributes(B,'pos'),
        node_size = 0,
        width = np.sqrt(line_capacities/vmax)*5,
        edge_color = 'black')
    
    L = nx.laplacian_matrix(B).A

    w, vv = np.linalg.eigh(L)
    v = -vv[:,1]
    minval = np.min(v)
    maxval = np.max(v)
    norm = MidpointNormalize(midpoint = 0,vmin = minval, vmax = maxval)  

    vmax = np.max(line_capacities)
    nx.draw(B,
            ax = ax[1,4],
            pos = nx.get_node_attributes(B,'pos'),
            node_color = norm(v),
            node_size = 30,
            width = np.sqrt(line_capacities/vmax)*1,
            edge_color = 'black',
            cmap = fiedler_primal_cmap)
    
    partition_levels = 3
    colors = np.array([boundary_dictionary_Europe[(u,v)] for u,v in B.edges()])
    width = np.array([boundary_width_Europe[(u,v)] for u,v in B.edges()])
    width[width==0] = 1.0
    width[width==5] = 5.
    if partition_levels < 4:
        level_3_indices = np.where(colors == 3)[0]
        colors[level_3_indices] = -1000
        width[level_3_indices] = 2.0

    cm = mpl.colors.ListedColormap([cut_cmap(i/nof_levels)
                                    for i in range(partition_levels)])
    cm.set_under(cmap_under, 1.0)
    edge_vmin = 0
    edge_vmax = partition_levels

    nx.draw_networkx_edges(B, 
                           pos = pos,
                           ax = ax[2,4],
                           width = width,
                           edge_color = colors,
                           edge_cmap = cm,
                           edge_vmin = edge_vmin,
                           edge_vmax = edge_vmax)
    
    # # Dual side
    B_dual = Europe_all_duals[0][0]
    dual_pos = nx.get_node_attributes(B_dual,'pos')

    line_capacities = [B_dual[u][v]['weight'] for u,v in B_dual.edges()]
    vmax = np.max(line_capacities)

    nx.draw(B_dual,
            ax = ax[0,5],
            pos = nx.get_node_attributes(B_dual,'pos'),
            node_size = 0,
            width = np.sqrt(line_capacities/vmax)*1,
            edge_color = 'black')


    L = nx.laplacian_matrix(B_dual).A

    _, vv = np.linalg.eigh(L)
    v = -vv[:,1]
    minval = np.min(v)
    maxval = np.max(v)
    norm = MidpointNormalize(midpoint = 0,vmin = minval, vmax = maxval)  

    vmax = np.max(line_capacities)
    nx.draw(B_dual,
            ax = ax[1,5],
            pos = nx.get_node_attributes(B_dual,'pos'),
            node_color = norm(v),
            node_size = 30,
            width = np.sqrt(line_capacities/vmax)*1,
            edge_color = 'black',
            cmap = fiedler_dual_cmap)
    
    partition_levels = 3
    colors = np.array([boundary_dictionary_Europe_dual[(u,v)] for u,v in B.edges()])
    width = np.array([boundary_width_Europe_dual[(u,v)] for u,v in B.edges()])
    width[width==0] = 1.0
    width[width==10] = 5.

    cm = mpl.colors.ListedColormap([cut_cmap(i/nof_levels)
                                    for i in range(partition_levels)])
    cm.set_under(cmap_under, 1.0)
    edge_vmin = 0
    edge_vmax = partition_levels


    nx.draw_networkx_edges(B, 
                           pos = pos,
                           ax = ax[2,5],
                           width = width,
                           edge_color = colors,
                           edge_cmap = cm,
                           edge_vmin = edge_vmin,
                           edge_vmax = edge_vmax)
    # Aesthetics
    for ax_r in ax[:, 2:].flatten():
        ax_r.set_rasterized(True)
    
    labels = [[r'\textbf{a}',r'\textbf{b}',r'\textbf{c}'],
              [r'\textbf{d}',r'\textbf{e}',r'\textbf{f}'],
              [r'\textbf{g}',r'\textbf{h}',r'\textbf{i}']]
    sm = plt.cm.ScalarMappable(cmap = plt.get_cmap('coolwarm'), norm=norm) 
    shift = 0.00

    for j in range(3):
        for i in range(6):
            ax[j, i].axis('off')
        for i in range(3):
            ax[j, 2 * i].text(0-shift, 1.02, labels[j][i],
                              fontsize=30, weight='bold',
                              verticalalignment = 'center',
                              transform = ax[j,2*i].transAxes)

    for i in range(2):
        for j in range(3):
            box = ax[j,i].get_position()
            box.x0 = box.x0 - 0.06
            box.x1 = box.x1 - 0.06
            ax[j,i].set_position(box)
            
    for i in range(2):
        for j in range(3):
            box = ax[j,i+2].get_position()
            box.x0 = box.x0 - 0.03
            box.x1 = box.x1 - 0.03
            ax[j,i+2].set_position(box)
            
    cbar_ax = fig.add_axes([0.31, 0.17, 0.005, 0.12])
    nof_levels = 3
    cmap = cut_cmap#PuBu_8.mpl_colormap
    cm = mpl.colors.ListedColormap([cmap(i/nof_levels) 
                                    for i in range(nof_levels)])
    cm.set_under(cmap_under,1.0)
    sm = plt.cm.ScalarMappable(cmap = cm) 
    sm._A = []
    cb = fig.colorbar(sm, cax=cbar_ax, ticks=[(i+.5)/nof_levels for i in range(nof_levels)])
    cb.ax.tick_params(labelsize = 36, width = 1.0)
    cb.ax.set_yticklabels([str(i+1) for i in range(nof_levels)], size = 20)
    cb.ax.set_title("Cut level", size=24)
        
    # Colorbar for small network example
    # #Primal side
    cbar_ax1 = fig.add_axes([0.055, 0.44, 0.005, 0.12])
    sm = plt.cm.ScalarMappable(cmap=fiedler_primal_cmap) 
    sm._A = []
    cb1 = fig.colorbar(sm, cax=cbar_ax1, ticks = [0,0.5,1])

    cb1.ax.set_yticklabels([r'$(\vec{v}_2)_i<0$', r'$0$',
                            r'$(\vec{v}_2)_i>0$'], size = 20)
    cb1.ax.yaxis.set_ticks_position('left')
    cb1.ax.tick_params(labelsize = 20, width = 1.0)
    
    # # Dual Side
    cbar_ax2 = fig.add_axes([0.31, 0.44, 0.005, 0.12])
    sm_dual = plt.cm.ScalarMappable(cmap=fiedler_dual_cmap) 
    sm_dual._A = []
    cb2 = fig.colorbar(sm_dual, cax=cbar_ax2, ticks=[0,0.5,1])
    cb2.ax.tick_params(labelsize = 36, width = 1.0)
    cb2.ax.set_yticklabels([r'$(\vec{v}^*_2)_i<0$',
                            r'$0$', r'$(\vec{v}^*_2)_i>0$'],size = 20)
    
    
    # Create brezier curves to show cut levels
    cut_path_one = np.asarray([[-.75, 0], [-.15, -.16], [0.061, -.849]])
    #ax[2,0].scatter(cut_path_one[:,0], cut_path_one[:,1])
    indent=0.07
    verts = [(xx[0] + dd, xx[1]) for xx in cut_path_one for dd in (-indent, 0, indent)][1:-1]
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) -1)
    path = Path(verts, codes)
    patch = mpl.patches.PathPatch(path, facecolor='none', lw=4, edgecolor=cut_cmap(0/nof_levels))
    ax[2, 0].add_patch(patch)    
    
    
    cut_path_one = np.asarray([[-.8, -0.6], [-.07, -.44]])
    #ax[2,0].scatter(cut_path_one[:,0], cut_path_one[:,1])
    indent=0.07
    verts = [(xx[0] + dd, xx[1]) for xx in cut_path_one for dd in (-indent, 0, indent)][1:-1]
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) -1)
    path = Path(verts, codes)
    patch = mpl.patches.PathPatch(path, facecolor='none', lw=4, edgecolor=cut_cmap(1/nof_levels),
                              zorder=3)

    ax[2, 0].add_patch(patch)   
    
    cut_path_one = np.asarray([[-.09, -0.260265], [0.318, -.016], [.553, .6766]])
    #ax[2,0].scatter(cut_path_one[:,0], cut_path_one[:,1])
    indent=0.07
    verts = [(xx[0] + dd, xx[1]) for xx in cut_path_one for dd in (-indent, 0, indent)][1:-1]
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) -1)
    path = Path(verts, codes)
    patch = mpl.patches.PathPatch(path, facecolor='none', lw=4, edgecolor=cut_cmap(1/nof_levels),
                              zorder=3)
    ax[2, 0].add_patch(patch)   
    
    # cut in dual
    cut_path_one = np.asarray([[.412, .72], [-.172, .7038], [.116, .1471],
                               [.17, -.464], [1.015, -.688]])
    indent=0.
    verts = [(xx[0] + dd, xx[1]) for xx in cut_path_one for dd in (-indent, 0, indent)][1:-1]
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) -1)
    path = Path(verts, codes)
    patch = mpl.patches.PathPatch(path, facecolor='none', lw=4, edgecolor=cut_cmap(1/nof_levels),
                              zorder=3)
    ax[2, 1].add_patch(patch) 
    
    
    cut_path_one = np.asarray([[-.7175, .3362], [.1141, .1382], [.7685, .1009]])
    indent=0.
    verts = [(xx[0] + dd, xx[1]) for xx in cut_path_one for dd in (-indent, 0, indent)][1:-1]
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(verts) -1)
    path = Path(verts, codes)
    patch = mpl.patches.PathPatch(path, facecolor='none', lw=4, edgecolor=cut_cmap(0/nof_levels),
                              zorder=3)
    ax[2, 1].add_patch(patch) 
    
    if save_fig:
        fig_path = "plots/primal_n_dual_spatial_networks.svg"
        
        fig.savefig(fig_path, bbox_inches='tight', transparent=True)
        
        fig.clear()
        plt.close(fig)
        
    else:
    
        plt.show()
        
    return


def algebraic_n_topological_connectivity(save_fig: bool = False, calc_eigvals: bool = False, use_sparse: bool = True,
                                         cut_lvl_grid: int = 2, cut_lvl_leafs:int = 3,
                                         grid_size: tuple = (11,7), axis_label_fontsize: int = 36):
    """Figure showing the connection between algebraic 
    and topological connecitivty for the square lattice and some leafs."""

    # List containing tuples with (label_graph, internal_label, file_with_graph)
    file_path_list = [('Acer platanoides', 'Maple', 'maple2c_80_graph.pklz'),
                      ('Schizolobium amazonicum', 'Schizolobium amazonicum',
                       'BronxA_125_binary_corrected_graph.pklz'),
                      ('Parkia nitida', 'Parkia nitida a',
                       'BronxA_115_a_binary_corrected_graph.pklz')]

    
    fig = plt.figure(figsize=(22, 8))
    gs = fig.add_gridspec(2, 4)
    ax = []
    ax.append([])
    ax[0].append(fig.add_subplot(gs[0, 0]))
    ax[0].append(fig.add_subplot(gs[0, 1]))
    ax[0].append(fig.add_subplot(gs[1, 0]))
    ax[0].append(fig.add_subplot(gs[1, 1]))
    
    ax.append(fig.add_subplot(gs[:, -2:]))
    

    sparse_str = ""
    if use_sparse:
        sparse_str = "_sparse"

    leaf_data_ls = list()

    for label_graph, name, file_r in file_path_list:

        with gzip.open(os.path.join(out_data_path, file_r), 'rb') as fh:
            data = pickle.load(fh)

            leaf_data_ls.append((label_graph, name, data))

    # N by M Grid
    N, M = grid_size

    G_grid = nx.generators.grid_2d_graph(N, M)
    pos = assign_pos(N, M)
    nx.set_node_attributes(G_grid, name='pos', values=pos)
    
    G_dual = nx.generators.grid_2d_graph(N-1, M-1)
    dual_pos = pos.copy()

    for nn in G_dual.nodes():
        dual_pos[nn] = np.array(dual_pos[nn]) + np.array([.5/N, .5/N])

    ## Create weak connections in dual graph
    edges = list()
    for ii in range(M-1):
        edges.append(((int(N/2), ii), (int(N/2), ii+1)))

    dual_edges = list()
    for ii in range(M-1):
        dual_edges.append(((int(N/2)-1, ii), (int(N/2), ii)))


    base = 1e0
    ww = {ee: base for ee in list(G_grid.edges())}
    edg = ww.copy()
    for e in edg.keys():
        ww[e[::-1]] = ww[e]
        
    kk = 1e1
    for ee in edges:
        ww[ee] = kk * base
        ww[ee[::-1]] = kk * base
    nx.set_edge_attributes(G_grid, ww, 'weight')
    
    ww = {ee: 1/base for ee in list(G_dual.edges())}
    edg = ww.copy()
    for e in edg.keys():
        ww[e[::-1]] = ww[e]
    k_dual = 1/kk
    for e in dual_edges:
        ww[e] = k_dual*base
        ww[e[::-1]] = k_dual*base
    nx.set_edge_attributes(G_dual, ww, 'weight')
    
    # Plot square grid
    nx.draw_networkx_nodes(G_grid, pos=pos, ax=ax[0][0], 
                           node_color='black', node_size=50)
    
    
    calc_G_dual = create_dual_from_graph(G_grid)
    _, _, boundaries_grid = get_hierarchy_levels_dual(G_grid, calc_G_dual, nof_levels=cut_lvl_grid,
                                                 use_median=False, networkx_fiedler=True)
    bound_dict_grid, bound_width_grid = get_boundary_infos(G_grid, boundaries_grid, cut_lvl=cut_lvl_grid)
    draw_actual_leaf(ax[0][0], G_grid, bound_dict_grid, bound_width_grid, cut_lvl_grid,
                     width_factor=3.)

    vein_weight_strong = np.logspace(base=10, start=np.log10(base),
                                     stop=6, num=50)

    ### Set up dictionary for edge weights
    w_dual = {ee: 1 for ee in list(G_dual.edges())}
    w_primal = {ee: 1 for ee in list(G_grid.edges())}
    edg = w_dual.copy()
    for e in edg.keys():
        w_dual[e[::-1]] = w_dual[e]
        w_primal[e[::-1]] = w_primal[e]

    ## Calculate Fiedler value
    Laplacian_spectrum_strong = np.zeros(
        (len(vein_weight_strong), len(list(G_dual.nodes()))))
    weight_sum_strong = np.zeros(len(vein_weight_strong))

    # todo don't run through list but actual edges
    count = 0
    for kk_r in tqdm(vein_weight_strong):
        kk_dual = 1/kk_r
        for ee in dual_edges:
            #w_dual[ee] = kk_dual
            #w_dual[ee[::-1]] = kk_dual
            G_dual.edges[ee]["weight"] = kk_dual
        for ee in edges:
            w_primal[ee] = kk_r
            #w_primal[ee[::-1]] = kk_r

        #nx.set_edge_attributes(G_dual, w_dual, 'weight')

        L_dual = nx.laplacian_matrix(G_dual).A
        ws, v = np.linalg.eigh(L_dual)
        Laplacian_spectrum_strong[count, :] = ws
        weight_sum_strong[count] = calc_dual_fiedler_estimate(len(G_dual.nodes())/2,
                                                              len(G_dual.nodes())/2,
                                                              edges,
                                                              w_primal)

        count += 1

    sc1 = ax[1].scatter(weight_sum_strong, Laplacian_spectrum_strong[:, 1], c=vein_weight_strong/base,
                        cmap='viridis', norm=mpl.colors.LogNorm(), label='square lattice (see \\textbf{a})')

    ax[1].plot(weight_sum_strong, weight_sum_strong, color="black", lw=2)

    # Plot leaves

    panel_label_ls = [r"\textbf{a}", r"\textbf{b}", r"\textbf{c}", r"\textbf{d}",
                      r"\textbf{e}", r"\textbf{f}", r"\textbf{g}"]
    graph_info_ls = list()

    for idx_r, leaf_data_r in enumerate(tqdm(leaf_data_ls)):
        label_leaf = leaf_data_r[0]
        name_leaf = leaf_data_r[1]
        test_leaf_data = leaf_data_r[2]

        graph_leaf_r = test_leaf_data[0]
        graph_dual_r = test_leaf_data[1]

        nodes_graph_leaf = len(graph_leaf_r.nodes())
        edges_graph_leaf = len(graph_leaf_r.edges())

        nodes_graph_dual = len(graph_dual_r.nodes())
        edges_graph_dual = len(graph_dual_r.edges())
        sum_weights_dual = np.sum(
            [ele for _, ele in nx.get_edge_attributes(graph_dual_r, 'weight').items()])

        graph_info_ls.append([name_leaf, nodes_graph_leaf, edges_graph_leaf,
                             nodes_graph_dual, edges_graph_dual, sum_weights_dual])

        # Plot leaf estimates
        marker_list = ['v', 's', '*', 'D', 'p']
        if calc_eigvals:
            lambda_2, estimate_lambda_2 = calculate_leaf_fielder_val_and_estimate(graph_leaf_r, graph_dual_r,
                                                                                  test_leaf_data[3], test_leaf_data[4],
                                                                                  use_sparse=use_sparse)

            if idx_r <= len(ax[0]) - 1:
                label_str = name_leaf + \
                    " (see {})".format(panel_label_ls[idx_r])
            else:
                label_str = name_leaf

            ax[1].scatter(estimate_lambda_2, lambda_2,
                          label=label_str, marker=marker_list[idx_r], s=100, linewidths=4)

        else:
            eig_load_path = (out_data_path + 
                "/{0}_fiedler_and_estimate{1}.pklz".format(name_leaf, sparse_str))
            if os.path.exists(eig_load_path):
                with gzip.open(eig_load_path, "rb") as fh_load:
                    _, lambda_2, estimate_lambda_2 = pickle.load(fh_load)

                if idx_r + 1 < len(ax[0]):
                    label_str = label_leaf + \
                        " (see {})".format(panel_label_ls[idx_r+1])
                else:
                    label_str = label_leaf

                # Check if only lambda_1 or entire spectrum
                if isinstance(lambda_2, np.ndarray):
                    lambda_2 = lambda_2[1]

                ax[1].scatter(estimate_lambda_2, lambda_2,
                              label=label_str, marker=marker_list[idx_r], s=100, linewidths=4)
            else:
                print("File '{}' does not exist!".format(eig_load_path))
                
        bound_dict, bound_width = get_boundary_infos(graph_leaf_r, test_leaf_data[4])
        if idx_r + 1 < len(ax[0]):
            draw_actual_leaf(ax[0][idx_r+1], graph_leaf_r,
                             bound_dict, bound_width, cut_lvl_leafs)
            ax[0][idx_r+1].set_xlabel(name_leaf, size=16)
            ax[0][idx_r+1].set_rasterized(True)
    
    # Aesthetics
    shift = .15
    ax0_label = panel_label_ls
    for idx_r, ax_r in enumerate(ax[0]):
        ax_r.axis('off')
        ax_r.text(0 - shift, 1, ax0_label[idx_r],
                  fontsize=30, weight='bold', verticalalignment='center',
                  transform=ax_r.transAxes)

    ax[1].text(0 - shift,
               1, r"\textbf{e}",
               fontsize=30,
               weight='bold',
               verticalalignment='center',
               transform=ax[1].transAxes)

    ax[1].legend(numpoints=None, fontsize=20, loc="upper left")
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')

    ax[1].set_ylabel('$\\lambda_2^*$', size=axis_label_fontsize, labelpad=10)
    ax[1].set_xlabel('$\\mu_2^*$', size=axis_label_fontsize, labelpad=10)
    ax[1].grid(visible=True)

    ax[1].tick_params(axis='both', which='major', labelsize=axis_label_fontsize)
    ax[1].tick_params(axis='both', which='minor', labelsize=axis_label_fontsize)
    ax[1].tick_params(axis='x', which='minor', labelsize=axis_label_fontsize)
    ax[1].xaxis.set_tick_params(which='both', labelbottom=True)
    ax[1].yaxis.set_tick_params(which='both', labelbottom=True)

    fig.tight_layout()
    
    cbar1 = fig.colorbar(sc1, ax=ax[1])
    cbar1.set_label("$w_1 / w_0$", size=axis_label_fontsize)
    cbar1.ax.tick_params(labelsize=axis_label_fontsize)

    
    cbar_ax = fig.add_axes([0.225, 0.225, 0.005, 0.12])#
    nof_levels = 3
    cmap = cut_cmap
    cm = mpl.colors.ListedColormap([cmap(i/nof_levels) for i in range(nof_levels)])
    cm.set_under(cmap_under,1.0)
    sm = plt.cm.ScalarMappable(cmap = cm) 
    sm._A = []
    cb = fig.colorbar(sm,cax = cbar_ax,ticks = [(i+.5)/nof_levels for i in range(nof_levels)])
    cb.ax.tick_params(labelsize = 36, width = 1.0)
    cb.ax.set_yticklabels([str(i+1) for i in range(nof_levels)],
                          size = 20)
    
    cb.ax.set_title('Cut level', size=20)
    
    
    if save_fig:
        fig_path = "plots/algebraic_n_topological_connectivity.pdf"
        fig.savefig(fig_path, bbox_inches='tight', transparent=True)

        fig.clear()
        plt.close(fig)

    else:
        plt.show()

    return


def primal_and_dual_communities_in_supply(save_fig: bool = True):
    """Figure showing that primal and dual communities arise naturally in supply
    networks in the European power system (obtain by the Energy System model PyPSA)
    and supply networks with stochastic sources (generated by 'stochastic_sources.py').

    Args:
        save_fig (bool, optional): _description_. Defaults to True.
    """
    
    # Load data
    path = raw_path + "/European_grid"
    with open(path + '/notime_elec_s_300_ec_lcopt_Co2/notime_elec_s_300_ec_lcopt_Co2L0.001-3H_opt_links.pickle', 'rb') as file:
        G_opt = pickle.load(file)
    
    with open(path + '/notime_elec_s_300_ec_lcopt_Co2/notime_elec_s_300_ec_lcopt_Co2L0.001-3H_opt_lines.pickle', 'rb') as file:
        F_opt = pickle.load(file)

    
    with open(path + '/notime_elec_s_300_ec_lcopt_Co2/notime_elec_s_300_ec_lcopt_Co2L0.40-3H_opt_links.pickle', 'rb') as file:
        G_nonopt = pickle.load(file)
    
    with open(path + '/notime_elec_s_300_ec_lcopt_Co2/notime_elec_s_300_ec_lcopt_Co2L0.40-3H_opt_lines.pickle', 'rb') as file:
        F_nonopt = pickle.load(file)

    with open(path + '/notime_elec_s_300_ec_lcopt_Co2/notime_elec_s_300_ec_lcopt_Co2L0.20-3H_opt_links.pickle', 'rb') as file:
        G_intermediate = pickle.load(file)
        
    with open(path + '/notime_elec_s_300_ec_lcopt_Co2/notime_elec_s_300_ec_lcopt_Co2L0.20-3H_opt_lines.pickle', 'rb') as file:
        F_intermediate = pickle.load(file)
    
    F_copy = F_opt.copy()
    for u,v in F_copy.edges():
        if F_copy[u][v]['weight'] == 0:
            F_opt.remove_edge(u,v)
            
    G_copy = G_opt.copy()
    for u,v in G_copy.edges():
        if G_copy[u][v]['weight'] == 0:
            G_opt.remove_edge(u,v)
            
    G_opt = nx.compose(F_opt, G_opt)

    F_copy = F_nonopt.copy()
    for u,v in F_copy.edges():
        if F_copy[u][v]['weight'] == 0:
            F_nonopt.remove_edge(u,v)
    G_copy = G_nonopt.copy()
    for u,v in G_copy.edges():
        if G_copy[u][v]['weight'] == 0:
            G_nonopt.remove_edge(u,v)
    G_nonopt = nx.compose(F_nonopt,G_nonopt)

    F_copy = F_intermediate.copy()
    for u,v in F_copy.edges():
        if F_copy[u][v]['weight'] == 0:
            F_intermediate.remove_edge(u,v)
    G_copy = G_intermediate.copy()
    for u,v in G_copy.edges():
        if G_copy[u][v]['weight'] == 0:
            G_intermediate.remove_edge(u,v)
    G_intermediate = nx.compose(F_intermediate, G_intermediate)
    
    sorted_GHG = np.loadtxt(path + '/GHG_reduction_levels_new.txt')
    primal_Fiedlers = np.loadtxt(path + '/primal_Fiedler_values_Europe_new.txt')
    dual_Fiedlers = np.loadtxt(path + '/dual_Fiedler_values_Europe_new.txt')

    with open(path + '/notime_elec_s_300_ec_lcopt_Co2/dual_networks.pickle', 'rb') as file:
        dual_networks = pickle.load(file)   
        
    G_opt_dual = dual_networks[-1]
    
    emission_to_fluctuations = np.loadtxt(path+'/fluctuations_and_carbon_levels.txt')
    
    # Load stochastic results  
    path_stoch = (raw_path + "/Stochastic_Sources")  
    with open(path_stoch + "/graphs_2_sources_minustwo_four_gamma9_run0.p", 'rb') as fh_leaf_graphs:
        leaf_graphs = pickle.load(fh_leaf_graphs)
    
    with open(path_stoch + "/all_primal_eigvals_2_sources_gamma0.9", 'rb') as fh_leaf_peigs:
        leaf_primal_eigs = pickle.load(fh_leaf_peigs)
        
    with open(path_stoch + "/all_dual_eigvals_2_sources_gamma0.9", 'rb') as fh_leaf_deigs:
        leaf_dual_eigs = pickle.load(fh_leaf_deigs)
        
    with open(path_stoch + "/dual_graph_0.p", 'rb') as fh_leaf_dual:
        leaf_dual = pickle.load(fh_leaf_dual)
        
    # Divide primal and dual Fiedler values by their traces
    fiedler_vals_primal_traced = leaf_primal_eigs[:, :, 1] / np.sum(leaf_primal_eigs, axis=2)
    fiedler_vals_dual_traced = list()
    for ii in range(leaf_dual_eigs.shape[0]):
        fiedler_vals_dual_traced.append([])
        for jj in range(leaf_dual_eigs.shape[1]):
            fiedler_vals_dual_traced[ii].append(leaf_dual_eigs[ii][jj][1] / 
                                                np.sum(leaf_dual_eigs[ii][jj]))
    
    # Plot it
    fig = plt.figure(figsize=(16, 10))
    
    gs = fig.add_gridspec(1, 2, width_ratios=[.6, 1.])
    gs_sub1 = gs[0].subgridspec(4, 2, hspace=0.3)
    gs_sub2 = gs[1].subgridspec(4, 1, hspace=0.6)
    
    ax_network_stoch_low = fig.add_subplot(gs_sub1[0, 0])
    ax_network_stoch_high = fig.add_subplot(gs_sub1[0, 1])
    ax_network_eu_conv = fig.add_subplot(gs_sub1[2, 0])
    ax_network_eu_renew = fig.add_subplot(gs_sub1[2, 1])
    
    ax_network_stoch_primal = fig.add_subplot(gs_sub1[1, 0])
    ax_network_stoch_dual = fig.add_subplot(gs_sub1[1, 1])
    ax_network_eu_primal = fig.add_subplot(gs_sub1[3, 0])
    ax_network_eu_dual = fig.add_subplot(gs_sub1[3, 1])
    
    ax_eigvals_stoch = fig.add_subplot(gs_sub2[:2])
    ax_eigvals_stoch_twin = ax_eigvals_stoch.twinx()
    ax_eigvals_eu = fig.add_subplot(gs_sub2[2:])
    ax_eigvals_eu_twin = ax_eigvals_eu.twinx()
    
    ax_network_ls = [ax_network_stoch_low,
                     ax_network_stoch_high,
                     ax_network_stoch_primal,
                     ax_network_stoch_dual,
                     ax_network_eu_conv,
                     ax_network_eu_renew,
                     ax_network_eu_primal,
                     ax_network_eu_dual]
    
    # Synthetic leafs
    ## low sigma_d
    
    leaf_low_stoch = leaf_graphs[-1].copy()
    leaf_high_stoch = leaf_graphs[0].copy()
    
    removed_edges = list()
    for ii in range(len(leaf_low_stoch.edges)):
        if leaf_low_stoch[list(leaf_low_stoch.edges())[ii][0]][list(leaf_low_stoch.edges())[ii][1]]['weight'] < 1e-8:
            removed_edges.append(list(leaf_low_stoch.edges())[ii])
    leaf_low_stoch.remove_edges_from(removed_edges)
    
    removed_edges = list()
    for ii in range(len(leaf_high_stoch.edges)):
        if leaf_high_stoch[list(leaf_high_stoch.edges())[ii][0]][list(leaf_high_stoch.edges())[ii][1]]['weight'] < 1e-8:
            removed_edges.append(list(leaf_high_stoch.edges())[ii])
    leaf_high_stoch.remove_edges_from(removed_edges)
    
    line_capacities_stoch_low = [leaf_low_stoch[uu][vv]['weight'] for uu, vv in leaf_low_stoch.edges()]
    line_capacities_stoch_high = [leaf_high_stoch[uu][vv]['weight'] for uu, vv in leaf_high_stoch.edges()]
    
    line_max_stoch = np.max(np.concatenate((line_capacities_stoch_high, line_capacities_stoch_low)))
    
    pos_stoch_low = nx.get_node_attributes(leaf_low_stoch, 'pos')
    
    nx.draw_networkx_edges(leaf_low_stoch, pos=pos_stoch_low, ax=ax_network_stoch_low,
                           width=np.sqrt(line_capacities_stoch_low/line_max_stoch)*7)
    
    fiedler_vec = nx.fiedler_vector(leaf_low_stoch)
    norm_fvec = MidpointNormalize(midpoint=0, vmin=np.min(fiedler_vec), vmax=np.max(fiedler_vec))
    
    nx.draw_networkx_nodes(leaf_low_stoch, pos=pos_stoch_low, ax=ax_network_stoch_primal,
                           node_size=5, node_color=norm_fvec(fiedler_vec),
                           cmap=fiedler_primal_cmap)
    nx.draw_networkx_edges(leaf_low_stoch, pos=pos_stoch_low, ax=ax_network_stoch_primal,
                           width=1)
    
    ## High sigma_d
    pos_stoch_high = nx.get_node_attributes(leaf_high_stoch, 'pos')
    nx.draw_networkx_edges(leaf_high_stoch, pos=pos_stoch_high, ax=ax_network_stoch_high,
                           width=np.sqrt(line_capacities_stoch_high/line_max_stoch)*7)
    
    fielder_vec_dual = nx.fiedler_vector(leaf_dual)
    norm_fvec_dual = MidpointNormalize(midpoint=0, vmax=np.max(fielder_vec_dual),
                                       vmin=np.min(fielder_vec_dual))
    
    dual_pos = nx.get_node_attributes(leaf_dual, 'pos')
    nx.draw_networkx_nodes(leaf_dual, pos=dual_pos,
                           ax=ax_network_stoch_dual, node_size=30,
                           node_color=norm_fvec_dual(fielder_vec_dual),
                           cmap=fiedler_dual_cmap)
    nx.draw_networkx_edges(leaf_dual, pos=dual_pos, ax=ax_network_stoch_dual,
                           width=.5)
    nx.draw_networkx_edges(leaf_high_stoch, pos=pos_stoch_high, width=3, alpha=.3,
                           ax=ax_network_stoch_dual)
    
    # EU grid
    # C02_red = 60%
    v = nx.fiedler_vector(G_nonopt)
    pos = nx.get_node_attributes(G_nonopt, 'pos')
    line_capacities = [G_nonopt[uu][vv]['weight'] for uu, vv in G_nonopt.edges()]
    edge_vmax = np.max(line_capacities)
    
    nx.draw_networkx_nodes(G_nonopt, pos, ax=ax_network_eu_conv, node_size=0)
    nx.draw_networkx_edges(G_nonopt, pos, ax=ax_network_eu_conv, 
                           width=np.sqrt(line_capacities/edge_vmax)*2)
    
    vmin = np.min(v)
    vmax = np.max(v)
    norm = MidpointNormalize(midpoint = 0,vmin = vmin, vmax = vmax)   

    nx.draw_networkx_nodes(G_nonopt, 
                            pos,
                            ax=ax_network_eu_primal, 
                            node_size=5,
                            node_color=norm(v),
                            cmap=fiedler_primal_cmap)

    nx.draw_networkx_edges(G_nonopt, 
                            pos,
                            ax=ax_network_eu_primal,
                            width = 1)
    
    # C02_red = 100%
    ## Primal graph
    pos = nx.get_node_attributes(G_opt, 'pos')
    line_capacities = [G_opt[uu][vv]['weight'] for uu, vv in G_opt.edges()]
    
    nx.draw_networkx_nodes(G_opt, pos, ax=ax_network_eu_renew, node_size=0)
    nx.draw_networkx_edges(G_opt, pos, ax=ax_network_eu_renew,
                           width=np.sqrt(line_capacities/edge_vmax)*2)
    
    ## Dual graph
    vv = nx.fiedler_vector(G_opt_dual)
    vmin = np.min(vv)
    vmax = np.max(vv)
    norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)
    dual_pos = nx.get_node_attributes(G_opt_dual, 'pos')
    nx.draw_networkx_nodes(G_opt_dual, dual_pos, ax=ax_network_eu_dual,
                           node_size=10,
                           node_color=norm(vv), cmap=fiedler_dual_cmap)
    nx.draw_networkx_edges(G_opt_dual, dual_pos, ax=ax_network_eu_dual, width=1)
    nx.draw_networkx_nodes(G_opt, 
                           pos,
                           ax=ax_network_eu_dual, 
                           node_size = 0)
    nx.draw_networkx_edges(G_opt, 
                           pos,
                           ax=ax_network_eu_dual,
                           width = 1,
                           alpha = 0.5,
                           edge_color = 'grey')
    
    # Plots of scans
    cmap_lines = plt.get_cmap('Blues')
    
    ## Stochastic networkx
    alphas = np.logspace(-2,4,50)
    nof_sources = 2
    K = 500
    sigmas = np.array(list(map(lambda x: K**2*(nof_sources-1)/
                               (nof_sources**2*(nof_sources*x+1)), alphas)))
    ln1 = ax_eigvals_stoch.plot(sigmas,np.quantile(fiedler_vals_primal_traced,
                                                   q=0.5,axis=0),color = cmap_lines(.8),
                                lw=3, label ='$\\lambda_2$')
    ax_eigvals_stoch.fill_between(sigmas, np.quantile(fiedler_vals_primal_traced, q=.5, axis=0),
                                  np.quantile(fiedler_vals_primal_traced, q=.75, axis=0),
                                  color=cmap_lines(.8), alpha=.5)
    ax_eigvals_stoch.fill_between(sigmas, np.quantile(fiedler_vals_primal_traced, q=.5, axis=0),
                                  np.quantile(fiedler_vals_primal_traced, q=.25, axis=0),
                                  color=cmap_lines(.8), alpha=.5)
    ax_eigvals_stoch.grid(True)
    ax_eigvals_stoch.set_ylim([10**(3/4)*1e-7,10**(1/4)*1e-5])
   
   
    ln2 = ax_eigvals_stoch_twin.plot(sigmas,np.quantile(fiedler_vals_dual_traced, 
                                                        q = 0.5,axis = 0),
                                     color = cmap_lines(.5),
                                     lw=3,
                                     label = '$\\lambda_2^*$')
    ax_eigvals_stoch_twin.fill_between(sigmas, np.quantile(fiedler_vals_dual_traced, q=.5, axis=0),
                                  np.quantile(fiedler_vals_dual_traced, q=.75, axis=0),
                                  color=cmap_lines(.5), alpha=.5)
    ax_eigvals_stoch_twin.fill_between(sigmas, np.quantile(fiedler_vals_dual_traced, q=.5, axis=0),
                                  np.quantile(fiedler_vals_dual_traced, q=.25, axis=0),
                                  color=cmap_lines(.5), alpha=.5)
    ax_eigvals_stoch_twin.set_ylim(bottom=3.1e-6, top=3.1e-3)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_eigvals_stoch.legend(lns,labs,fontsize = 22, loc=6)
    
    ## EU grid
    ax_eigvals_eu.scatter(sorted_GHG, primal_Fiedlers, s=200,
                          label='$\\lambda_2$', color=cmap_lines(.8))
    ax_eigvals_eu.plot(sorted_GHG, primal_Fiedlers, color=cmap_lines(.8), lw=3.)
    
    ax_eigvals_eu_twin.scatter(sorted_GHG, dual_Fiedlers, s=200, 
                               label='$\\lambda_2^*$', color=cmap_lines(.5))
    
    ax_eigvals_eu_twin.plot(sorted_GHG, dual_Fiedlers, 
                            color=cmap_lines(.5), lw=3)
    
    lines, labels = ax_eigvals_eu.get_legend_handles_labels()
    lines2, labels2 = ax_eigvals_eu_twin.get_legend_handles_labels()
    ax_eigvals_eu.legend(lines+lines2,labels+labels2, fontsize = 22, 
                         loc = 6)
    
    # Aesthetics
    for ax_r in ax_network_ls:
        ax_r.axis('off')
    
    ax_network_stoch_low.set_title("$\\sigma_D^2 = " +
                                   "{0:.1f}$".format(sigmas[-1]),
                                   size=20)
    ax_network_stoch_high.set_title("$\\sigma_D^2 = " +
                                    "{0:.1f} \\cdot 10^4$".format(sigmas[0]/10**4),
                                    size=20)
    
    ax_network_eu_conv.set_title("CO$_{2}$ red. $=60\\%$", size=20)
    ax_network_eu_renew.set_title("CO$_{2}$ red. $=100\\%$", size=20)
    
    ax_eigvals_eu.set_xlabel("CO$_2$ emission reduction", size=26)
    ax_eigvals_eu.set_ylabel("$\\lambda_2$", size=26)
    ax_eigvals_eu_twin.set_ylabel("$\\lambda_2^*$", size=26)
    
    ax_eigvals_stoch.set_xscale('log')
    ax_eigvals_stoch_twin.set_xscale('log')
    for ax_r in [ax_eigvals_eu, ax_eigvals_eu_twin,
                 ax_eigvals_stoch, ax_eigvals_stoch_twin]:
        ax_r.set_yscale('log')
        ax_r.tick_params(axis = 'both', which = 'major', labelsize = 20)
        ax_r.tick_params(axis = 'both', which = 'minor', labelsize = 20)
        ax_r.tick_params(axis = 'x', which = 'minor', labelsize = 20)
        
    ax_eigvals_stoch.set_xlabel("Fluctuation strength $\\sigma_D^2$", size=26, labelpad=-5)
    ax_eigvals_stoch.set_ylabel("$\\lambda_2 / \\text{trace}(\\mathbf{L})$", size=26)
    ax_eigvals_stoch_twin.set_ylabel("$\\lambda_2^* / \\text{trace}(\\mathbf{L}^*)$", size=26)
    
    panel_labels = ["$\\textbf{a}$", "$\\textbf{b}$", "$\\textbf{c}$", "$\\textbf{d}$",
                    "$\\textbf{e}$","$\\textbf{f}$","$\\textbf{g}$","$\\textbf{h}$",
                    "$\\textbf{i}$","$\\textbf{j}$","$\\textbf{k}$",]
    
    ax_all_ls = [ax_network_stoch_low, ax_network_stoch_high, ax_network_stoch_primal,
                 ax_network_stoch_dual, ax_eigvals_stoch,
                 ax_network_eu_conv, ax_network_eu_renew, ax_network_eu_primal,
                 ax_network_eu_dual, ax_eigvals_eu]
    index_label = 0
    for kk, ax_r in enumerate(ax_all_ls):
        ax_r.text(-.1, 1.075, panel_labels[kk], fontsize=24, weight='bold',
                  transform=ax_r.transAxes)
        index_label += 1
        
    for ax_r in [ax_eigvals_eu, ax_eigvals_stoch]:
        ax_r.grid(True)

    
    fig.tight_layout()
    fig.subplots_adjust(bottom=.15, hspace=.5)
    
    # Emission vs fluctuating 
    ax_last_pos = ax_eigvals_eu.get_position()
    ax_fluct = fig.add_axes((ax_last_pos.x0, ax_last_pos.y0 - .11,
                             ax_last_pos.x1 - ax_last_pos.x0 - .01, .04))
    
    emission_to_fluctuations_sorted = emission_to_fluctuations[np.argsort(emission_to_fluctuations[:, 0])]
    ax_fluct.fill_between(emission_to_fluctuations_sorted[:,0],
                          emission_to_fluctuations_sorted[:,1], where=emission_to_fluctuations[:, 1] > 0, 
                          color='white')
    lower_fluct = emission_to_fluctuations[np.argmin(abs(emission_to_fluctuations[:,0] - .59)), 1] 
    high_fluct = emission_to_fluctuations[np.argmin(abs(emission_to_fluctuations[:,0] - 1.0)), 1]
    ax_fluct.set_facecolor('white')    
    ax_fluct.set_yticks([np.round(lower_fluct, decimals=2)])
    ax_fluct.set_ylim(bottom=0)
    ax_fluct.set_xlim(left=min(emission_to_fluctuations[:,0]), right=max(emission_to_fluctuations[:,0]))
    ax_fluct.set_xlabel("Share of fluctuating renewables", size=20)
    ax_eigvals_eu.set_xlim(left=.59, right=1.01)
    
    ax_fluct_twin = ax_fluct.twinx()
    ax_fluct_twin.fill_between(emission_to_fluctuations_sorted[:,0],
                          emission_to_fluctuations_sorted[:,1], where=emission_to_fluctuations[:, 1] > 0, 
                          color='grey')
    ax_fluct_twin.set_yticks([np.round(high_fluct, decimals=2)])
    
    ax_fluct.set_ylim(bottom=0)
    ax_fluct_twin.set_ylim(bottom=0)
    
    for ax_r in [ax_fluct, ax_fluct_twin]:
        ax_r.spines['top'].set_visible(False)
        ax_r.tick_params(axis = 'both', which = 'major', labelsize = 20,
                         width=1, length=5)
        ax_r.set_xlim(left=.59, right=1.)
        ax_r.set_xticks([])
        
    if save_fig:
        fig_path = "plots/primal_dual_comms_in_supplynetworks.pdf"
        
        fig.savefig(fig_path, transparent=True, bbox_inches='tight')
        
        fig.clear()
        plt.close(fig)
        
    else:
        plt.show()
    
    return


def all_figures():
    
    # Different structural pattern separate networks and increase network robustness
    connection_communities_network_robustness(save_fig=True)
    
    # Primal and dual communities and hierarchies in spatial networks
    primal_and_dual_communities_in_spatial_networks(save_fig=True)
    
    # Algebraic and Topological Connectivity in the dual graph of synthetic and real-world networks
    algebraic_n_topological_connectivity(save_fig=True)
    
    # Primal and dual communities emerge naturally in optimal supply networks.
    primal_and_dual_communities_in_supply(save_fig=True)
    