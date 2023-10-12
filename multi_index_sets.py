# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:57:36 2023

@author: D. Loukrezis
"""

import itertools
import math
import numpy as np
from scipy.special import comb


def setsize(N, w):
    """
    Returns the number of PCE polynomials of total-degree basis given the 
    number of dimensions N and the maximum polynomial degree w.
    """
    return int(comb(N+w-1, N-1))    


def td_set_recursive(N, w, rows):
    """
    Help function for the recursive computation of the total-degree 
    multiindices.
    """
    if N == 1:
        subset = w*np.ones([rows, 1])
    else:
        if w == 0:
            subset = np.zeros([rows, N])
        elif w == 1:
            subset = np.eye(N)
        else:
            # initialize submatrix
            subset = np.empty([rows, N])
            
            # starting row of submatrix
            row_start = 0
            
            # iterate by polynomial order and fill the multiindex submatrices
            for k in range(0, w+1):
                
                # number of rows of the submatrix
                sub_rows = setsize(N-1, w-k)
                
                # update until row r2
                row_end = row_start + sub_rows - 1
                
                # first column
                subset[row_start:row_end+1, 0] = k*np.ones(sub_rows)
                
                # subset update --> recursive call
                subset[row_start:row_end+1, 1:] = td_set_recursive(N-1, w-k, 
                                                              sub_rows)
                                                                     
                # update row indices
                row_start = row_end + 1
    
    return subset


def td_multiindex_set(N, w):
    """
    Returns the total-degree multiindex set for N parameters and maximum 
    polynomial degree w
    
    **Inputs**
    
    * **N** (`int`):
        Number of parameters/dimensions.
        
    * **w** (`int`):
        Maximum polynomial degree.
        
    **Output:**
        ndarray (KxN, K being the number of PCE terms) with the total-degree 
        multiindices
    """
    
    # size of the total degree multiindex set
    td_size = int(comb(N+w, N))
    
    # initialize total degree multiindex set
    midx_set = np.empty([td_size, N])
    
    # starting row
    row_start = 0
    
    # iterate by polynomial order
    for i in range(0, w+1):
        
        # compute number of rows
        rows = setsize(N, i)
        
        # update up to row r2
        row_end = rows + row_start - 1
        
        # recursive call 
        midx_set[row_start:row_end+1, :] = td_set_recursive(N, i, rows)
        
        # update starting row
        row_start = row_end + 1
        
    return midx_set.astype(int)


def tp_multiindex_set(N, w):
    """
    Returns the tensor-product multiindex set for N parameters and maximum 
    polynomial degree w
    
    **Inputs**
    
    * **N** (`int`):
        Number of parameters/dimensions.
        
    * **w** (`int`):
        Maximum polynomial degree.
        
    **Output:**
        ndarray (KxN, K being the number of PCE terms) with the tensor-product 
        multiindices
    """
    orders = np.arange(0, w+1, 1).tolist()
    if N == 1:
        midx_set = np.array(list(map(lambda el:[el], orders)))
    else:
        midx = list(itertools.product(orders, repeat=N))
        midx = [list(elem) for elem in midx]
        midx_sums = [int(math.fsum(midx[i])) for i in range(len(midx))]
        midx_sorted = sorted(range(len(midx_sums)), 
                             key=lambda k: midx_sums[k])
        midx_set = np.array([midx[midx_sorted[i]] for i in range(len(midx))])   
    return midx_set.astype(int)