# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:00:08 2023

@author: D. Loukrezis
"""

#%% imports

# general
import numpy as np
import openturns as ot

# model and distribution
from scim_model import scim_param
from scim_distributions import dist_joint

import matplotlib.pyplot as plt
plt.rcParams.update(
    {
     'text.usetex':True,
     'figure.figsize': [6.5, 4.5],
     'figure.dpi': 300,
     'font.size': 20,
     'font.family': 'serif',
     "lines.linewidth": 3.0
    }
)
legend_font_size=12

# list of seeds
seed_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# num. samples
nsamples = 1500

# strings to be used for saving data
folder_str = 'scim_data_DL/'
txt_str =  '.txt'
inputs_str = 'inputs'
outputs_str = 'outputs'
seed_str = '_seed='
 
if __name__ == '__main__':
    
    # iterate over seeds
    for seed in seed_list:
        # set seed
        ot.RandomGenerator.SetSeed(seed)
        seed_str_now = seed_str + str(seed)
        
        data_in  = np.array(dist_joint.getSample(nsamples))
        
        save_inputs = folder_str + inputs_str + seed_str_now + txt_str
        print(save_inputs)
        np.savetxt(save_inputs, data_in)
        
        torques = []
        plt.figure()
        for i in range(nsamples):
            print(i)
            state, time = scim_param(
                r_s=data_in[i,0],
                r_r=data_in[i,1],
                l_m=data_in[i,2],
                l_sigs=data_in[i,3],
                l_sigr=data_in[i,4],
                j_rotor=data_in[i,5],
                amplitude=data_in[i,6],
                f_grid=data_in[i,7],
                a=data_in[i,8],
                b=data_in[i,9],
                c=data_in[i,10],
                j_load=data_in[i,11]
            ) 
            torques.append(state[1])
            plt.plot(time, state[1], '-y')
        torques=np.array(torques)
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Torque (Nm)')
        plt.tight_layout()
        plt.xlim(0,60)
        plt.xticks([0,20,40,60])
        
        save_outputs = folder_str + outputs_str + seed_str_now + txt_str
        print(save_outputs)
        print()
        np.savetxt(save_outputs, torques)
        
        
        