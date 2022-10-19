#!usr/bin/env python
# -*- coding: utf-8 -*-

"""Misc tools used by multiple scripts"""

import numpy as np

def assign_pos(mm: int, nn: int) -> np.ndarray:
    """Assign position in 1 by 1 square to each lattice point of m x n square lattice."""
    
    pos = {}
    for i in range(mm):
        for j in range(nn):
            pos[i,j] = np.array([i*float(1/mm), j*float(1/nn)])
            
    return pos
