# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:20:29 2023

@author: D. Loukrezis
"""

import numpy as np
import openturns as ot


# parameter distributions 
dist_Rs = ot.Normal(2.9338, 2.9338*0.05)
dist_Rr = ot.Normal(1.355, 1.355*0.05)
dist_Lm = ot.Normal(143.75e-3, 143.75e-3*0.05) 
dist_Lsigs = ot.Normal(5.87e-3, 5.87e-3*0.05)
dist_Lsigr = ot.Normal(5.87e-3, 5.87e-3*0.05)
dist_Jrotor = ot.Normal(0.0011, 0.0011*0.05)
dist_Ampl = ot.Uniform(0.9, 1.0)
dist_Fgrid = ot.Uniform(49.5, 50.5)
dist_A = ot.Normal(1e-3, 1e-3*0.05)
dist_B = ot.Normal(1e-3, 1e-3*0.05)
dist_C = ot.Normal(1e-3, 1e-3*0.05)
dist_Jload = ot.Normal(1e-3, 1e-3*0.05)

# joint distribution
dist_joint = ot.ComposedDistribution([dist_Rs, dist_Rr, dist_Lm, 
                                        dist_Lsigs, dist_Lsigr, dist_Jrotor, 
                                        dist_Ampl, dist_Fgrid, dist_A, 
                                        dist_B, dist_C, dist_Jload])
