# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:30:53 2023

@author: z004dp6c
"""

import numpy as np


# test with vector-valued QoI
def simply_supported_beam(x, Ns=9):
    '''
    Parameters
    ----------
    x : numpy array
        x[0]: beam width b (m)
        x[1]: beam height h (m)
        x[2]: beam length L (m)
        x[3]: Young's modulus E (MPa)
        x[4]  (uniform) load P (N/m)
    Ns : integer
        Longitudinal coordinates along the beam: s_i = i*L/Ns, i=1,...,Ns. 
        The default is 9.

    Returns
    -------
    V(s) : (negative) displacement of the beam at the longitudinal coordinates
    '''
    b = x[0]
    h = x[1]
    L = x[2]
    E = x[3]
    P = x[4]
    s = np.array([i*L/(Ns+1) for i in range(1, Ns+1)])
    V = -P*s*(L**3 - 2*L*s**2 + s**3) / (2*E*b*h**3)
    return V



