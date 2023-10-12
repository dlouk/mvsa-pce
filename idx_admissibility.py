# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 11:30:20 2017

@author: D. Loukrezis

Index forward/backward neighbors and admissibility for monotone sets.
"""

import numpy as np


def admissible_neighbors(index, index_set):
    """Given an index and a monotone index set, find admissible neighboring
    indices"""
    for_neighbors = forward_neighbors(index)
    # find admissible neighbors
    for_truefalse = [is_admissible(fn, index_set) for fn in for_neighbors]
    adm_neighbors = np.array(for_neighbors)[for_truefalse].tolist()
    return adm_neighbors


def is_admissible(index, index_set):
    """Given an index and a monotone index set, check index admissibility"""
    back_neighbors = backward_neighbors(index)
    for ind_b in back_neighbors:
        if ind_b not in index_set:
            return False
    return True


def forward_neighbors(index):
    """Given a multiindex, return its forward neighbors as a list of
    multiindices, e.g. (2,1) --> (3,1), (2,2)"""
    N = len(index)
    for_neighbors = []
    for i in range(N):
        index_tmp = index[:]
        index_tmp[i] = index_tmp[i] + 1
        for_neighbors.append(index_tmp)
    return for_neighbors


def backward_neighbors(index):
    """Given a multiindex, return its backward neighbors as a list of
    multiindices, e.g. (2,2) --> (1,2), (2,1)"""
    N = len(index)
    back_neighbors = []
    for i in range(N):
        index_tmp = index[:]
        if index_tmp[i] > 0:
            index_tmp[i] = index_tmp[i] - 1
            back_neighbors.append(index_tmp)
    return back_neighbors

