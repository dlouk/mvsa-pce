# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:00:08 2023

@author: z004dp6c
"""

#%% imports

# general
import numpy as np
import openturns as ot

# model and distribution
from beam_model import simply_supported_beam 
from beam_distributions import dist_joint

# list with desired number of outputs
# !!! num. outputs will be value-1, i.e., [9, 49, 99, ...] !!!
num_outputs_list =  [10, 50, 100, 500, 1000]

# list of seeds
seed_list = np.arange(0,10).tolist()

# num. samples
nsamples = 1500

# strings to be used for saving data
folder_str = 'beam_data/'
txt_str =  '.txt'
nsamples_str = '_nsamples=' + str(nsamples)
inputs_str = 'inputs'
outputs_str = 'outputs'
seed_str = '_seed='
nout_str =  '_nout='
 
if __name__ == '__main__':
    
    # iterate over seeds
    for seed in seed_list:
        print()
        # set seed
        ot.RandomGenerator.SetSeed(seed)
        seed_str_now = seed_str + str(seed)
        
        data_in  = np.array(dist_joint.getSample(nsamples))
        
        
        save_inputs = folder_str + inputs_str + seed_str_now + nsamples_str +\
                      txt_str
        print(save_inputs)
        np.savetxt(save_inputs, data_in)
        
        
        # iterate over number of outputs
        for nout in num_outputs_list:    
            nout_str_now =  nout_str + str(nout)
            data_out = np.array([simply_supported_beam(din[:5], Ns=nout) 
                                 for din in data_in])
        
            save_outputs = folder_str + outputs_str + seed_str_now +\
                           nsamples_str + nout_str_now + txt_str
            print(save_outputs)
            np.savetxt(save_outputs, data_out)
       
        

        
        
        
            



