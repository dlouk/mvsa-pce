# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:22:45 2023

@author: D. Loukrezis
"""

import openturns as ot
import numpy as np
from pce import PolynomialChaosExpansion
from idx_admissibility import admissible_neighbors


class SensitivityAdaptivePCE():
    def __init__(self, pdf, exp_design_in, exp_design_out):
        
        # case without given pdf: identify PDF from data
        if pdf == None:
            chaos_algo_data = ot.FunctionalChaosAlgorithm(exp_design_in, 
                                                          exp_design_out)
            chaos_algo_data.run()
            self.pdf = chaos_algo_data.getDistribution()
        else:    
            self.pdf = pdf
        
        self.exp_design_in = exp_design_in
        self.exp_design_out = exp_design_out
        
        # initialize a 1st order PCE
        self.pce = PolynomialChaosExpansion(self.pdf, 
                                            self.exp_design_in, 
                                            self.exp_design_out)
        self.pce.compute_coefficients()
        
        # initialize active multi-index set with zero-multi-index 
        self.active_multi_indices = [self.pce.multi_index_set[0]]
        
        # initialize admissible multi-index set with all 1st order multi-indices
        self.admissible_multi_indices = self.pce.multi_index_set[1:]
        
        # compute aggregated partial variance of each admissible multi-index
        admissible_coefficients = self.pce.coefficients[1:].tolist()
        aggregated_admissible_coefficients = np.sum(
                             np.abs(admissible_coefficients), axis=1)
        
        # find admissible multi-index with maximum aggregated coefficients
        # and remove it from the admissible multi-index set 
        help_index = np.argmax(aggregated_admissible_coefficients)
        max_admissible_multi_index = self.admissible_multi_indices.pop(help_index)
        
        # update the active multi-index set
        self.active_multi_indices.append(max_admissible_multi_index)
    
            
    def construct_adaptive_basis(self, max_condition_number=1e2):
        while True:
            if self.pce.condition_number > max_condition_number:
                #print('Design matrix not sufficiently well-conditioned.')
                #print('Adaptive basis construction ends here.')
                break
            
            # find new admissible multi-indices
            new_admissible_multi_indices = admissible_neighbors(
                                                self.active_multi_indices[-1],
                                                self.active_multi_indices)
            
            # check if the basis terms are more than the training data points
            num_terms = len(self.active_multi_indices) + \
                        len(self.admissible_multi_indices) +\
                        len(new_admissible_multi_indices)
            if num_terms >= len(self.pce.exp_design_inputs):
                break
            
            # update the admissible multi-index set
            self.admissible_multi_indices += new_admissible_multi_indices
            
            # compute PCE for full multi-index set
            all_multi_indices = self.active_multi_indices + \
                                                self.admissible_multi_indices
                                                                        
            self.pce.set_multi_index_set(all_multi_indices)
            self.pce.construct_basis()
            self.pce.compute_coefficients()
            
            # starting index for admissible multi-indices
            idx = len(self.active_multi_indices)  
            # get admissible coefficients
            admissible_coefficients = self.pce.coefficients[idx:].tolist()
            
            # compute aggregated coefficients of each admissible multi-index
            aggregated_admissible_coefficients = np.sum(
                             np.abs(admissible_coefficients), axis=1)
            
            # find admissible multi-index with maximum aggregated partial variance,
            # add it to active multi-index set and remove it from admissible 
            # multi-index set
            help_index = np.argmax(aggregated_admissible_coefficients)
            max_admissible_multi_index = self.admissible_multi_indices.pop(
                                                                help_index)
            self.active_multi_indices.append(max_admissible_multi_index)
            
    def construct_active_pce(self):
        pce = PolynomialChaosExpansion(self.pdf, 
                                        self.exp_design_in, 
                                        self.exp_design_out)
        pce.set_multi_index_set(self.active_multi_indices)
        pce.construct_basis()
        pce.compute_coefficients()
        return pce
    
    def construct_augmented_pce(self):
        pce = PolynomialChaosExpansion(self.pdf, 
                                        self.exp_design_in, 
                                        self.exp_design_out)
        pce.set_multi_index_set(self.active_multi_indices + 
                                self.admissible_multi_indices)
        pce.construct_basis()
        pce.compute_coefficients()
        return pce      
    
    def construct_reduced_augmented_pce(self, max_condition_number=1e2):
        # compute augmented pce
        pce = self.construct_augmented_pce()
        while True:
            # exit if the condition number is acceptable
            if pce.condition_number <= max_condition_number and\
                len(pce.multi_index_set) <= len(pce.exp_design_inputs):
                break
            # remove single- and multi-index with minimum contribution
            idx_min = np.argmin(np.sum(np.abs(pce.coefficients), axis=1))
            pce.multi_index_set.pop(idx_min)
            pce.single_index_set.pop(idx_min)
            # compute pce basis and coefficients with reduced multi-index set
            pce.construct_basis()
            pce.compute_coefficients()
        return pce
        
        
        
