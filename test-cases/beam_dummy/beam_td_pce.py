# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:43:18 2023

@author: z004dp6c
"""

#%% imports

# general 
import numpy as np
import openturns as ot
import time
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# model and distribution
from beam_model import simply_supported_beam
from beam_distributions import dist_joint_normal

# data 
from beam_data_generation import num_outputs_list, seed_list, nsamples
from beam_data_generation import folder_str, txt_str, nsamples_str, inputs_str
from beam_data_generation import outputs_str, seed_str, nout_str

# pce
import sys
sys.path.append('../../')
from multi_index_sets import td_multiindex_set
from pce import PolynomialChaosExpansion


#%% main
if __name__ == '__main__':
    # define td, nout, and ntrain
    for td in [2,3]:
        print('TD:', td)
        
        for nout in [10, 100, 1000]:
            print('nout:', nout)
        
            for ntrain in [50, 100, 150]:
                print('ntrain:', ntrain)
                
                # lists for saving results
                vector_rmse_list = []
                avg_rmse_list = []
                time_list = []
                mean_list = []
                variance_list = []
                std_list = []
                
                #%% iterate over seeds
                for seed in seed_list:
                    print('seed:', seed)
                    
                    # get inputs for given seed
                    fname_inputs = folder_str + inputs_str + seed_str +\
                                   str(seed) + nsamples_str + txt_str
                    input_data = np.log(np.genfromtxt(fname_inputs))
                    
                    # get outputs for given seed and size of output
                    fname_outputs = folder_str + outputs_str + seed_str +\
                                    str(seed) + nsamples_str + nout_str +\
                                        str(nout) + txt_str
                    output_data = np.genfromtxt(fname_outputs)
                    
                    # test data will always be the last 1000 data points
                    test_in = input_data[500:, :]
                    test_out = output_data[500:, :]
                            
                    train_in = input_data[:ntrain, :]
                    train_out = output_data[:ntrain, :]
                            
                    #%% generate PCE 
                    t0 = time.time()
                    pce = PolynomialChaosExpansion(dist_joint_normal, 
                                                   train_in, 
                                                   train_out)
                    
                    # compute a total-degree multi-index set
                    td_set = td_multiindex_set(pce.num_inputs, td).tolist()
                    
                    # set new PCE multi-index set 
                    pce.set_multi_index_set(td_set)
                    
                    # construct PCE basis based on the new multi-index set
                    pce.construct_basis()
                    
                    # compute the coefficients
                    pce.compute_coefficients()
                    t1 = time.time()                                
                                                    
                    # make predictions and compute errors wrt to the true model 
                    # outputs
                    pce_predictions = pce.predict(test_in)
                    vector_rmse = root_mean_squared_error(
                        test_out, 
                        pce_predictions, 
                        multioutput='raw_values')
                    avg_rmse = root_mean_squared_error(
                        test_out, 
                        pce_predictions, 
                        multioutput='uniform_average')
                    
                    # compute mean, variance, and std
                    mean = pce.compute_mean()
                    variance = pce.compute_variance()
                    std = np.sqrt(variance)
                    
                    # update lists
                    vector_rmse_list.append(vector_rmse)
                    avg_rmse_list.append(avg_rmse)
                    time_list.append(t1-t0)
                    mean_list.append(mean)
                    variance_list.append(variance)
                    std_list.append(std)
                                
                    
                # save vector rmse per nout, ntrain
                np.savetxt('beam_results/vector_rmse/' + 'td=' + str(td) +\
                           '_vector_rmse' + '_nout=' + str(nout) +\
                               '_ntrain=' + str(ntrain) + '.txt', 
                               vector_rmse_list)
                
                # save average rmse per nout, ntrain 
                np.savetxt('beam_results/avg_rmse/' + 'td=' + str(td) +\
                           '_avg_rmse' + '_nout=' + str(nout) + '_ntrain=' + 
                           str(ntrain) + '.txt', avg_rmse_list)
                
                # save computation_time per nout, ntrain 
                np.savetxt('beam_results/computation_time/' + 'td=' +\
                           str(td) + '_computation_time' + '_nout=' +\
                               str(nout) + '_ntrain=' + str(ntrain) + '.txt', 
                               time_list)
                    
                # save mean per nout, ntrain 
                np.savetxt('beam_results/mean/' + 'td=' + str(td) + '_mean' +\
                           '_nout=' + str(nout) + '_ntrain=' + str(ntrain) +\
                           '.txt', mean_list)
                    
                # save variance per nout, ntrain 
                np.savetxt('beam_results/variance/' + 'td=' + str(td) +\
                           '_variance' + '_nout=' + str(nout) +\
                           '_ntrain=' + str(ntrain) + '.txt', variance_list)
                    
                # save std per nout, ntrain 
                np.savetxt('beam_results/std/' + 'td=' + str(td) + '_std' +\
                           '_nout=' + str(nout) + '_ntrain=' + str(ntrain) +\
                           '.txt', std_list)
                
                