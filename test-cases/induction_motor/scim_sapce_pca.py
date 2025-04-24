# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:26:18 2023

@author: D. Loukrezis
"""

#%% imports

# general 
import numpy as np
import openturns as ot
import time
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# distribution
from scim_distributions import dist_joint

# pce
import sys
sys.path.append('../../')
from sapce import SensitivityAdaptivePCE

# list of seeds
seed_list = [1] #np.arange(1,11).astype(int).tolist()

# strings for loading data
data_folder = 'scim_data/'
input_file = 'inputs_seed'
output_file = 'outputs_seed'
nsamples = 1500
 

if __name__ == '__main__':
    #%% iterate over seeds
    for seed in seed_list:
        # get inputs for given seed
        fname_inputs = data_folder + input_file + str(seed) + '.txt'
        input_data = np.genfromtxt(fname_inputs)
        
        
        # get outputs for given seed
        fname_outputs = data_folder + output_file + str(seed) + '.txt'
        output_data = np.genfromtxt(fname_outputs)
        
        # test data will always be the last 1000 data points
        test_in = input_data[500:, :]
        test_out = output_data[500:, :]
        
        
        #%% iterate over training data set sizes
        num_training_data = np.arange(20,101,20).astype(int).tolist()
        num_training_data = num_training_data +\
                            np.arange(150,501,50).astype(int).tolist()
        
        # lists for saving results
        mse_list = []
        rmse_list = []
        time_list = []
        
        for ntrain in num_training_data:
            
            # training data
            train_in = input_data[:ntrain, :]
            train_out = output_data[:ntrain, :]
            
            # dimension reduction of the output
            pca = PCA(n_components=10)
            pca.fit(train_out)
            train_out_reduced = pca.transform(train_out)
            
            # compute sensitivity-adaptive PCE
            sapce = SensitivityAdaptivePCE(dist_joint, 
                                            train_in, 
                                            train_out_reduced)
            t0 = time.time()
            sapce.construct_adaptive_basis(max_condition_number=1e2)
            #active_pce = sapce.construct_active_pce()
            augmented_pce = sapce.construct_reduced_augmented_pce()
            t1 = time.time()
            
            # make predictions and compute errors wrt to the true model 
            # outputs
            pce_predictions_reduced = augmented_pce.predict(test_in)
            pce_predictions = pca.inverse_transform(pce_predictions_reduced)
            mse = mean_squared_error(test_out, pce_predictions)
            rmse = np.sqrt(mse)
            
            # update lists
            time_list.append(t1-t0)
            mse_list.append(mse)
            rmse_list.append(rmse)
            
            # on-screen output
            print('Seed', seed)
            print('Ntrain:', ntrain)
            print('Computation time:', t1-t0)
            print('MSE:', mse)
            print('RMSE:', rmse)
            print()
        
        # save results
        to_save = np.array([num_training_data, time_list, mse_list, 
                            rmse_list]).T
        folder_name = 'scim_results/'
        save_name = 'sapce_pca' + '_seed' + str(seed) + '.txt'
        np.savetxt(folder_name + save_name, to_save)
