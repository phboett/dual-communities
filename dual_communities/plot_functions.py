#!usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as _np
import networkx as _nx
import matplotlib as _mpl
from matplotlib import pyplot as _plt


def draw_actual_leaf(ax, leaf, boundary_dict, boundary_width, nof_levels,
                     width_factor: float = 1., cmap=_plt.get_cmap('plasma_r'),
                     cmap_under='darkgray', use_bound_width: bool = False):
    """Draw the leaf in ax"""

    pos = _nx.get_node_attributes(leaf, 'pos')

    if use_bound_width:
        width_edges = _np.array([boundary_width[(u, v) * width_factor] for u, v in leaf.edges()])
    else:
        width_edges = width_factor   
    
    cm = _mpl.colors.ListedColormap(
        [cmap(i/nof_levels) for i in range(nof_levels)])
    cm.set_under(cmap_under, 1.0)
    edge_vmin = 0
    edge_vmax = nof_levels
    
    boundary_edges = [key for key, xx in boundary_dict.items() if xx >=0]
    subgraph = leaf.edge_subgraph(boundary_edges)
    for (uu, vv), ele in boundary_dict.items():
        if ele >= 0:
            subgraph[uu][vv]['edge_color'] = ele
    boundary_colors = [subgraph[uu][vv]['edge_color'] 
                       for uu, vv in subgraph.edges()]
    
    _nx.draw_networkx_edges(leaf, pos=pos, ax=ax,
                           width=width_edges, edge_color=cmap_under)
    
    _nx.draw_networkx_edges(subgraph,
                           pos=_nx.get_node_attributes(subgraph, 'pos'),
                           ax=ax,
                           width=4*width_edges,
                           edge_color=boundary_colors,
                           edge_cmap=cm,
                           edge_vmin=edge_vmin,
                           edge_vmax=edge_vmax)

    return