# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:43:18 2023

@author: D. Loukrezis
"""

#%% imports

# general 
import numpy as np
import openturns as ot
import time
from sklearn.metrics import mean_squared_error

# distribution
from waveguide_distributions import dist_joint

# data 
from waveguide_data_generation import num_outputs_list, seed_list, nsamples
from waveguide_data_generation import folder_str, inputs_str, outputs_str
from waveguide_data_generation import seed_str, nout_str, txt_str, nsamples_str

# pce
import sys
sys.path.append('../../')
from multi_index_sets import td_multiindex_set
from pce import PolynomialChaosExpansion


if __name__ == '__main__':
    
    #%% iterate over seeds
    for seed in seed_list:
        # get inputs for given seed
        fname_inputs = folder_str + inputs_str + seed_str + str(seed) +\
                       nsamples_str + txt_str
        input_data = np.genfromtxt(fname_inputs)
        
        #%% iterate over output sizes
        for nout in num_outputs_list:
            # get outputs for given seed and size of output
            fname_outputs = folder_str + outputs_str + seed_str + str(seed) +\
                           nsamples_str + nout_str + str(nout) + txt_str
            output_data = np.genfromtxt(fname_outputs)
            
            # test data will always be the last 1000 data points
            test_in = input_data[500:, :]
            test_out = output_data[500:, :]
            
            #%% iterate over total degrees
            total_degrees = [2,3]
            for td in total_degrees:
                
                # lists for saving results
                mse_list = []
                rmse_list = []
                time_list = []
                
                #%% iterate over training data set sizes
                num_training_data = np.arange(20,101,20).astype(int).tolist()
                num_training_data = num_training_data +\
                                     np.arange(150,501,50).astype(int).tolist()
                for ntrain in num_training_data:
                    train_in = input_data[:ntrain, :]
                    train_out = output_data[:ntrain, :]
                
                    #%% generate PCE 
                    pce = PolynomialChaosExpansion(dist_joint, 
                                                   train_in, 
                                                   train_out)
                    
                    # compute a total-degree multi-index set
                    td_set = td_multiindex_set(pce.num_inputs, td).tolist()
                    
                    t0 = time.time()
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
                    mse = mean_squared_error(test_out, pce_predictions)
                    rmse = np.sqrt(mse)
                    
                    # update lists
                    time_list.append(t1-t0)
                    mse_list.append(mse)
                    rmse_list.append(rmse)
                    
                    # on-screen output
                    print('Seed', seed)
                    print('Num. outputs:', nout)
                    print('Total degree:', td)
                    print('Ntrain:', ntrain)
                    print('Computation time:', t1-t0)
                    print('MSE:', mse)
                    print('RMSE:', rmse)
                    print()
                
                # save results
                to_save = np.array([num_training_data, time_list, mse_list, 
                                    rmse_list]).T
                folder_name = 'waveguide_results/'
                save_name = 'tdpce_p=' + str(td) + nout_str + str(nout) +\
                            seed_str + str(seed) + txt_str
                np.savetxt(folder_name + save_name, to_save)