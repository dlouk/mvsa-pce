# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:06:31 2023

@author: D. Loukrezis
"""

#%%
import numpy as np
import openturns as ot
from waveguide_model import debye2_broadband
from waveguide_distributions import dist_joint

#%% for data generation

num_outputs_list = [10, 50]

# list of seeds
seed_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# num. samples
nsamples = 1500


# strings to be used for saving data
folder_str = 'waveguide_data/'
txt_str =  '.txt'
nsamples_str = '_nsamples=' + str(nsamples)
inputs_str = 'inputs'
outputs_str = 'outputs'
seed_str = '_seed='
nout_str =  '_nout='


if __name__ == '__main__':
    
    # iterate over seeds
    for seed in seed_list:
        # set seed
        ot.RandomGenerator.SetSeed(seed)
        seed_str_now = seed_str + str(seed)
        
        data_in  = np.array(dist_joint.getSample(nsamples))
        
        
        save_inputs = folder_str + inputs_str + seed_str_now + nsamples_str +\
                      txt_str
        print(save_inputs)
        np.savetxt(save_inputs, data_in)
        
        fmin = 6
        fmax = 30
        # iterate over number of outputs
        for nout in num_outputs_list:    
            nout_str_now =  nout_str + str(nout)
            data_out = np.array([debye2_broadband(fmin=fmin, fmax=fmax,
                fsamples=nout, width=din[0], height=din[1], fill_l=din[2], 
                offset=din[3], epss1=din[4], epss2=din[5], eps8=din[6], 
                tau_eps_const1=din[7], tau_eps_const2=din[8], mues1=din[9], 
                mues2=din[10], mue8=din[11], tau_mue_const1=din[12], 
                tau_mue_const2=din[13], res='dB'
                ) for din in data_in])
        
            save_outputs = folder_str + outputs_str + seed_str_now +\
                           nsamples_str + nout_str_now + txt_str
            print(save_outputs)
            np.savetxt(save_outputs, data_out)
       
        

        
        
        
            



