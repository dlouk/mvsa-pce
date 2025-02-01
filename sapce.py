# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:22:45 2023

@author: z004dp6c
"""

import openturns as ot
import numpy as np
from pce import PolynomialChaosExpansion
from idx_admissibility import admissible_neighbors
from multi_index_sets import td_multiindex_set

class SensitivityAdaptivePCE():
    def __init__(self, pdf, exp_design_in, exp_design_out, 
                 max_partial_degree=10):
        
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

        self.max_partial_degree = max_partial_degree
        
        # initialize a 1st order PCE
        num_inputs = self.pdf.getDimension()
        td1_set = td_multiindex_set(num_inputs, 1).tolist()
        self.pce = PolynomialChaosExpansion(self.pdf, 
                                            self.exp_design_in, 
                                            self.exp_design_out)
        self.pce.set_multi_index_set(td1_set)
        self.pce.construct_basis()
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
    
            
    def construct_adaptive_basis(self, max_condition_number=1e2, termination_info=True):
        while True:
            # T1: condition number termination
            if self.pce.condition_number > max_condition_number:
                if(termination_info):
                    print("Adaptive basis construction terminated:" 
                        + " design matrix not sufficiently well-conditioned.")
                break
            
            # find new admissible multi-indices
            new_admissible_multi_indices = admissible_neighbors(
                                                self.active_multi_indices[-1],
                                                self.active_multi_indices)
            
            # T2: maximum partial polynomial degree termination
            # Use only admissible multi indices where the max. degree is 
            # smaller than the 'max_partial_degree' parameter 
            for idx, adm_multi_Indcs in reversed(list(enumerate(new_admissible_multi_indices))):
                for adm_multi_Indx in adm_multi_Indcs:
                    if adm_multi_Indx > self.max_partial_degree:
                        new_admissible_multi_indices.pop(idx)
            # When there are no admissible neighbors, where the partial degree
            # is lower or equals than 'max_partial_degree' -> terminate  
            if [self.max_partial_degree]*self.pce.num_inputs in self.active_multi_indices: 
                if len(new_admissible_multi_indices) == 0:
                    if(termination_info):
                        print("Adaptive basis construction terminated:" 
                        + " maximum partial degree reached.")
                    break
            
            # T3: basis cardinality vs. ED size termination
            # check if the basis terms are more than the training data points
            num_terms = len(self.active_multi_indices) + \
                        len(self.admissible_multi_indices) +\
                        len(new_admissible_multi_indices)
            if num_terms >= len(self.pce.exp_design_inputs):
                if(termination_info):
                    print("Adaptive basis construction terminated:" 
                        + " basis cardinality reached experimental design size.")
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
        # re-order multi-indices and coefficients
        # coeffs = pce.coefficients
        # midx = np.array(pce.multi_index_set)
        # coeffs_aggr = np.sum(np.abs(coeffs), axis=1)
        # order_idx = np.flip(np.argsort(coeffs_aggr))
        # midx_ord = midx[order_idx, :].tolist()
        # pce.set_multi_index_set(midx_ord)
        # pce.construct_basis()
        # pce.compute_coefficients()
        return pce
    
    def construct_coefficient_pruned_pce(self, cr=1e-8):
        # compute augmented pce
        pce = self.construct_augmented_pce()        

        # OLD coefficient pruning
        # cr = 1e-8
        # prune_coeffs = np.vectorize(
        #     lambda x: 0 if np.abs(x) < cr else x
        # )
        # pce.coefficients = prune_coeffs(pce.coefficients)
        #
        ## At this point we now have a set of pce coefficients and multi-indices
        ## where most of the coefficients are zero. But the zero-associated
        ## multi-indices are still in the mult_set_list, what is irritating,
        ## when checking the highest polynomial degree used in the PCE approximation.
        #
        # NEW coefficient pruning
        for i, elem in reversed(list(enumerate(np.sum(pce.coefficients,axis=1)))):
            if np.abs(elem) < cr:
                pce.multi_index_set.pop(i)
                pce.coefficients = np.delete(pce.coefficients, i, axis=0)

        # update pce multi indices and single indices
        pce.set_multi_index_set(pce.multi_index_set)
        # we need to construct a new basis to update 
        # the associated paramters in the pce object.
        # If one does not construct a new basis the design matrix
        # will be built up on the old, "unpruned" basis size.
        # ->    So we basically update the basis and 
        #       the number of basis polynomials used.
        pce.construct_basis()
        
        return pce
        
    
    # def construct_ordered_augmented_pce(self):
    #     # compute augmented pce
    #     pce = self.construct_augmented_pce()
    #     # get coefficients and multi-index set
    #     coeffs = pce.coefficients
    #     midx = np.array(pce.multi_index_set)
    #     # compute aggregated coefficients
    #     coeffs_aggr = np.sum(np.abs(coeffs), axis=1)
    #     # find ordering according from maximum to minimum aggr. coeff.
    #     order_idx = np.flip(np.argsort(coeffs_aggr))
    #     # order multiindices and coefficients according to aggr.coeff.
    #     midx_ord = midx[order_idx, :].tolist()
    #     pce.set_multi_index_set(midx_ord)
    #     pce.construct_basis()
    #     pce.compute_coefficients()
    #     return pce
        
        
