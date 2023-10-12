# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 21:59:24 2023

@author: D. Loukrezis
"""

import openturns as ot

# nominal values of waveguide model
width_nom  = 30.0
height_nom = 3.0
fill_l_nom = 7.0
offset_nom = 10.0
epss1_nom  = 2.0
epss2_nom  = 2.2
eps8_nom   = 1.0
mues1_nom  = 2.0
mues2_nom  = 3.0
mue8_nom  = 1.0
tau_eps_const1_nom = 1.0
tau_eps_const2_nom = 1.1
tau_mue_const1_nom = 1.0
tau_mue_const2_nom = 2.0

nominal_list = [width_nom, height_nom, fill_l_nom, offset_nom, epss1_nom, 
                epss2_nom, eps8_nom, mues1_nom, mues2_nom, mue8_nom, 
                tau_eps_const1_nom, tau_eps_const2_nom, tau_mue_const1_nom, 
                tau_mue_const2_nom]

# maximum variation around nominal value
max_var = 0.05 

# marginal PDFs
distributions = []
for nom in nominal_list:
    distributions.append(ot.Uniform(nom-nom*max_var, nom+nom*max_var))

# joint pdf
dist_joint = ot.ComposedDistribution(distributions)
