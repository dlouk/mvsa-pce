# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:51:07 2018

@authors: D. Loukrezis
"""

import numpy as np


def compute_design_matrix(basis, ed_in):
    """
    Computes the PCE design matrix for a given PCE basis and experimental 
    design (inputs only)
    
    Inputs
    ------
    basis: PCE basis (OpenTURNS object)
    ed_in: experimental design (2D numpy array)
    
    Outputs
    -------
    D: 2d numpy array of size (ED_size x Basis_size)
    """
    ed_size = ed_in.shape[0]
    basis_size = len(basis)
    D = np.zeros([ed_size, basis_size])
    for j in range(basis_size):
        D[:,j] = np.array(basis[j](ed_in)).flatten()
    return D


def transform_multi_index_set(midx_set, enum_func):
    """
    Transforms a multi-index set to a single-index set using an OpenTURNS 
    enumerate function
    
    midx_set: multi-index set (e.g. [[0,0],[1,0]])
    enum_func: enumerate function
    """
    sidx = []
    for idx in midx_set:
        sidx.append(enum_func.inverse(idx))
    return sidx
