# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 18:26:39 2023

@author: D. Loukrezis
"""

import openturns as ot
import numpy as np


class PolynomialChaosExpansion():
    '''
    Class for a flexible OpenTURNS-based polynomial chaos expansion
    '''
    def __init__(self, pdf, exp_design_in, exp_design_out):
        
        # case without given pdf: identify PDF from data
        if pdf == None:
            chaos_algo_data = ot.FunctionalChaosAlgorithm(exp_design_in, 
                                                          exp_design_out)
            chaos_algo_data.run()
            self.pdf = chaos_algo_data.getDistribution()
        else:    
            self.pdf = pdf
        self.num_inputs = pdf.getDimension()
        self.num_outputs = exp_design_out.shape[1]
        self.num_samples = exp_design_out.shape[0]
        self.exp_design_inputs = exp_design_in
        self.exp_design_outputs = exp_design_out
        
        # As initialization, compute a PCE of 1st order polynomials only, i.e., 
        # a total degree PCE with total degree equal to 1. It is inexpensive to 
        # compute, we will most probably have to compute it either way, and we 
        # get some helpful OpenTURNS functions to be used later, most 
        # importantly, the Transformation function.
        
        # get enumerate function for single and multi-indices
        self.enumerate_function = ot.LinearEnumerateFunction(self.num_inputs)
        
        # assign the correct polynomials based on the input distribution
        self.polynomial_collection = [
                  ot.StandardDistributionPolynomialFactory(
                      self.pdf.getMarginal(i))
                  for i in range(self.num_inputs)]
        
        # get a general product basis, to be used for the construction of 
        # specific PCE bases, i.e., given specific index sets
        self.product_basis = ot.OrthogonalProductPolynomialFactory(
                        self.polynomial_collection, self.enumerate_function)
        
        # compute total degree PCE with total degree equal to 1
        total_degree = 1 
        basis_size = self.enumerate_function.getStrataCumulatedCardinal(
                                                                total_degree) 
        adaptive_strategy = ot.FixedStrategy(self.product_basis, basis_size) 
        projection_strategy = ot.LeastSquaresStrategy()
        chaos_algo = ot.FunctionalChaosAlgorithm(exp_design_in, exp_design_out, 
                                                 pdf, adaptive_strategy, 
                                                 projection_strategy)
        chaos_algo.run()
        chaos_result = chaos_algo.getResult()
        
        # get the transformation function to be applied for standardization
        # of the input data
        self.transformation = chaos_result.getTransformation()
        
        # get the single and multi-indices corresponding to the 1st order PCE
        self.single_index_set = np.array(chaos_result.getIndices()).tolist()
        self.multi_index_set = [list(self.enumerate_function(idx)) 
                                for idx in self.single_index_set]
        
        # get polynomial basis corresponding to the 1st order PCE
        self.basis = chaos_result.getReducedBasis()
        self.num_polynomials = self.basis.getSize()
        
        # get PCE coefficients corresponding to the 1st order PCE
        self.coefficients = np.array(chaos_result.getCoefficients())
        
        # compute design matrix corresponding to the 1st order PCE and the 
        # given experimental design
        self.design_matrix = np.zeros([self.num_samples, 
                                       self.num_polynomials])
        exp_design_inputs_tf = self.transformation(self.exp_design_inputs)
        for j in range(self.num_polynomials):
            self.design_matrix[:, j] = np.array(
                self.basis[j](exp_design_inputs_tf)).flatten()
        
    def set_multi_index_set(self, multi_index_set):
        self.multi_index_set = multi_index_set
        self.single_index_set = [self.enumerate_function.inverse(idx) 
                                  for idx in self.multi_index_set]
        
    def set_single_index_set(self, single_index_set):
        self.single_index_set = single_index_set
        self.multi_index_set = [list(self.enumerate_function(idx)) 
                                     for idx in self.single_index_set]
        
    def construct_basis(self):
        self.basis = self.product_basis.getSubBasis(self.single_index_set)
        self.num_polynomials = self.basis.getSize()
        
    def set_exp_design(self, exp_design_in, exp_design_out):
        if self.num_outputs != exp_design_out.shape[1]:
            raise ValueError('Output dimensions do not agree!')
        self.exp_design_inputs = exp_design_in
        self.exp_design_outputs = exp_design_out 
        self.num_samples = exp_design_out.shape[0]
        
    def evaluate_basis(self, design_in):
        n_samples, n_inputs = np.shape(design_in)
        if n_inputs != self.num_inputs:
            raise ValueError('Input dimensions do not agree!')
        # transform input data
        design_in_tf = np.array(self.transformation(design_in))
        # compute basis evaluation matrix
        eval_matrix = np.empty([n_samples, self.num_polynomials])
        for j in range(self.num_polynomials):
            eval_matrix[:,j] = np.array(self.basis[j](design_in_tf)).flatten()
        return eval_matrix
    
    def compute_design_matrix(self):
        self.design_matrix = self.evaluate_basis(self.exp_design_inputs)
    
    def compute_coefficients(self):
        # compute design matrix
        self.compute_design_matrix()
        # compute PCE coefficients using least squares regression
        self.coefficients, _, _, singular_values = np.linalg.lstsq(
                                                     self.design_matrix, 
                                                     self.exp_design_outputs,
                                                     rcond=None)
        self.condition_number = singular_values[0] / singular_values[-1]
    
    def predict(self, design_in):
        eval_matrix = self.evaluate_basis(design_in)
        return eval_matrix.dot(self.coefficients)
    
    def compute_mean(self):
        return self.coefficients[0]
    
    def compute_variance(self):
        return np.sum(np.square(self.coefficients[1:]), axis=0)
    
    def compute_sobol_first(self):
        '''
        Computes the first order Sobol indices. For vector-valued outputs, the
        Sobol indices are computed elementwise, , i.e., 1 index per element of 
        the output.
        '''
        sobol_f = np.empty([self.num_inputs, self.num_outputs])
        variance = self.compute_variance()
        # remove zeroth multi-index from multi-index set
        midx_minus_0 = np.delete(self.multi_index_set, 0, axis=0)
        for i in range(self.num_inputs):
            # remove i-th column from multi-index set without 0
            midx_minus_i = np.delete(midx_minus_0, i, axis=1)
            # get the rows with all indices equal to zero
            row_sum = np.sum(midx_minus_i, axis=1)
            zero_rows = np.asarray(np.where(row_sum==0)).flatten() + 1
            partial_variance = np.sum(np.square(self.coefficients[zero_rows]),
                                      axis=0)
            sobol_f[i,:] = partial_variance / variance
        return sobol_f
    
    def compute_sobol_total(self):
        '''
        Computes the total order Sobol indices. For vector-valued outputs, the
        Sobol indices are computed elementwise, i.e., 1 index per element of 
        the output.
        '''
        sobol_t = np.empty([self.num_inputs, self.num_outputs])
        variance = self.compute_variance()
        for i in range(self.num_inputs):
            # we want all multi-indices where the i-th index is NOT zero  
            idx_column_i = np.array(self.multi_index_set)[:, i]
            non_zero_rows = np.asarray(np.where(idx_column_i!=0)).flatten()
            partial_variance = np.sum(
                                np.square(self.coefficients[non_zero_rows]),
                                axis=0)
            sobol_t[i,:] = partial_variance / variance
        return sobol_t
    
    def compute_generalized_sobol_first(self):
        '''
        Computes the generalized first order Sobol indices. For scalar-valued 
        outputs, the generalized Sobol indices coincide with the standard Sobol
        indices.
        '''
        # compute variance and first order Sobol indices (elementwise)
        variance = self.compute_variance()
        sobol_f = self.compute_sobol_first()
        # retrieve elementwise partial variances
        partial_variances = sobol_f*variance
        # compute aggregated variance (scalar value)
        aggregated_variance = np.sum(variance)
        # compute aggregated partial variance per input 
        # 1d array with length equal to num_inputs
        aggregated_partial_variances = np.sum(partial_variances, axis=1)
        # compute generalized first order Sobol indices
        sobol_f_gen = aggregated_partial_variances / aggregated_variance
        return sobol_f_gen

    def compute_generalized_sobol_total(self):
        '''
        Computes the generalized total order Sobol indices. For scalar-valued 
        outputs, the generalized Sobol indices coincide with the standard Sobol
        indices.
        '''
        # compute variance and total order Sobol indices (elementwise)
        variance = self.compute_variance()
        sobol_t = self.compute_sobol_total()
        # retrieve elementwise partial variances
        partial_variances = sobol_t*variance
        # compute aggregated variance (scalar value)
        aggregated_variance = np.sum(variance)
        # compute aggregated partial variance per input 
        # 1d array with length equal to num_inputs
        aggregated_partial_variances = np.sum(partial_variances, axis=1)
        # compute generalized total order Sobol indices
        sobol_t_gen = aggregated_partial_variances / aggregated_variance
        return sobol_t_gen



