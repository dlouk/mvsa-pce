# -*- coding: utf-8 -*-
"""
Created on Sat May 13 00:49:53 2023

@author: D. Loukrezis
"""

import time
import numpy as np
import gym_electric_motor as gem
import openturns as ot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = [6 , 4]
plt.rcParams['figure.dpi'] = 200 # 200 e.g. is really fine, but slower
plt.rcParams.update({'font.size': 16})

def parameterize_three_phase_grid(amplitude, frequency, initial_phase):
    """
    This nested function allows to create a function of time, which returns 
    the momentary voltage of the three-phase grid.
    The nested structure allows to parameterize the three-phase grid by 
    amplitude (as a fraction of the DC-link voltage), frequency (in Hertz), 
    and initial phase (in degree).
    """

    omega = frequency * 2 * np.pi  # 1/s
    phi = 2 * np.pi / 3  # phase offset
    phi_initial = initial_phase * 2 * np.pi / 360

    def grid_voltage(t):
        u_abc = [
            amplitude * np.sin(omega * t + phi_initial),
            amplitude * np.sin(omega * t + phi_initial - phi),
            amplitude * np.sin(omega * t + phi_initial + phi)
         ]
        return u_abc
    
    return grid_voltage


def scim_param(
        r_s=2.9338, # [r_s] = Ohm, stator resistance
        r_r=1.355, # [r_r] = Ohm, rotor resistance
        l_m=143.75e-3, # [l_m] = H, main inductance
        l_sigs=5.87e-3, # [l_sigs] = H, stator-side stray inductance
        l_sigr=5.87e-3, # [l_sigs] = H, rotor-side stray inductance
        j_rotor=0.0011, # [j_rotor] = kg/m^2, moment of intertia of rotor
        amplitude=1.0, # amplitude of DC-link voltage
        f_grid=50, # Hz, grid frequency 
        a=0.0, # Nm: Constant Load Torque coefficient (static friction)
        b=0.0, # Nm: Linear Load Torque coefficient (sliding friction)
        c=0.0, # Nm: Quadratic Load Torque coefficient (air friction) 
        j_load=1e-6,
        p=2 # [p] = --, nb of pole pairs
        ):
    
    # define motor arguments
    motor_parameter = dict(
        r_s=r_s, # [r_s] = Ohm, stator resistance
        r_r=r_r, # [r_r] = Ohm, rotor resistance
        l_m=l_m, # [l_m] = H, main inductance
        l_sigs=l_sigs, # [l_sigs] = H, stator-side stray inductance
        l_sigr=l_sigr, # [l_sigs] = H, rotor-side stray inductance
        p=p, # [p] = --, nb of pole pairs
        j_rotor=j_rotor # [j_rotor] = kg/m^2, moment of intertia of rotor
    )

    # Create the environment
    env = gem.make(
        # Choose the squirrel cage induction motor (SCIM) with continuous-control-set
        "Cont-CC-SCIM-v0",
        #
        motor=dict(motor_parameter=motor_parameter),
        #
        load=gem.physical_systems.PolynomialStaticLoad(
            dict(a=a, b=b, c=c, j_load=j_load)
           
        ),

        # Define the numerical solver for the simulation
        ode_solver="scipy.ode",

        # Define which state variables are to be monitored concerning limit 
        # violations
        # "()" means, that limit violation will not necessitate an env.reset()
        constraints=(),

        # Set the sampling time
        tau=1e-4
    )

    tau = env.physical_system.tau
    limits = env.physical_system.limits

    # reset the environment such that the simulation can be started
    (state, reference) = env.reset()

    # We define these arrays in order to save our simulation results in them
    # Initial state and initial time are directly inserted
    STATE = np.transpose(np.array([state * limits]))
    TIME = np.array([0])

    # Use the previously defined function to parameterize a three-phase grid with 
    # an amplitude of x % of the DC-link voltage and a frequency of 50 Hertz
    u_abc = parameterize_three_phase_grid(amplitude=amplitude, 
                                          frequency=f_grid, 
                                          initial_phase=0)

    # Set a time horizon to simulate, in this case 60 ms
    time_horizon = 0.2
    step_horizon = int(time_horizon / tau)
    for idx in range(step_horizon):
        # calculate the time of this simulation step
        time = idx * tau

        # apply the voltage as given by the grid
        (state, reference), reward, done, _ = env.step(u_abc(time))
        #(state, reference), reward, done, _ = env.step()
        
        # save the results of this simulation step
        STATE = np.append(STATE, np.transpose([state * limits]), axis=1)
        TIME = np.append(TIME, time)

    # convert the timescale from s to ms
    TIME *= 1e3
    
    return STATE, TIME

 # r_s=2.9338, # [r_s] = Ohm, stator resistance
 # r_r=1.355, # [r_r] = Ohm, rotor resistance
 # l_m=143.75e-3, # [l_m] = H, main inductance
 # l_sigs=5.87e-3, # [l_sigs] = H, stator-side stray inductance
 # l_sigr=5.87e-3, # [l_sigs] = H, rotor-side stray inductance
 # j_rotor=0.0011, # [j_rotor] = kg/m^2, moment of intertia of rotor
 # amplitude=1.0, # amplitude of DC-link voltage
 # f_grid=50, # Hz, grid frequency 
 # a=0.0, # Nm: Constant Load Torque coefficient (static friction)
 # b=0.0, # Nm: Linear Load Torque coefficient (sliding friction)
 # c=0.0, # Nm: Quadratic Load Torque coefficient (air friction) 
 # j_load=1e-6,


if __name__ == '__main__':
    
    # parameter distributions 
    var = 0.05
    dist_Rs = ot.Normal(2.9338, 2.9338*var)
    dist_Rr = ot.Normal(1.355, 1.355*var)
    dist_Lm = ot.Normal(143.75e-3, 143.75e-3*var) 
    dist_Lsigs = ot.Normal(5.87e-3, 5.87e-3*var)
    dist_Lsigr = ot.Normal(5.87e-3, 5.87e-3*var)
    dist_Jrotor = ot.Normal(0.0011, 0.0011*var)
    dist_Ampl = ot.Uniform(0.9, 1.0)
    dist_Fgrid = ot.Uniform(49.5, 50.5)
    dist_A = ot.Normal(1e-3, 1e-3*var)
    dist_B = ot.Normal(1e-3, 1e-3*var)
    dist_C = ot.Normal(1e-3, 1e-3*var)
    dist_Jload = ot.Normal(1e-3, 1e-3*var)
   
    # joint distribution
    dist_joint = ot.ComposedDistribution([dist_Rs, dist_Rr, dist_Lm, 
                                          dist_Lsigs, dist_Lsigr, dist_Jrotor, 
                                          dist_Ampl, dist_Fgrid, dist_A, 
                                          dist_B, dist_C, dist_Jload])

    seed_list=[1,2,3,4,5,6,7,8,9,10]
    for seed in [1]:#seed_list:
        print("seed: {}".format(seed))
        time_start=time.time()
        # generate test data
        ot.RandomGenerator.SetSeed(seed) # set seed for reproducibility
        Ntest = 10
        test_in = np.array(dist_joint.getSample(Ntest))
        torques = []
        for i in range(Ntest):
            print(i)
            state, time_x = scim_param(r_s=test_in[i,0],
                                     r_r=test_in[i,1],
                                     l_m=test_in[i,2],
                                     l_sigs=test_in[i,3],
                                     l_sigr=test_in[i,4],
                                     j_rotor=test_in[i,5],
                                     amplitude=test_in[i,6],
                                     f_grid=test_in[i,7],
                                     a=test_in[i,8],
                                     b=test_in[i,9],
                                     c=test_in[i,10],
                                     j_load=test_in[i,11]) 
            torques.append(state[1])
        torques=np.array(torques)
        
        time_end=time.time()
        print("Execution time: {}s".format(time_end - time_start))
        
        plt.figure()
        for i in range(Ntest):
            plt.plot(time_x, torques[i,:])
            
        # np.savetxt('design_inputs_seed{}_n{}.txt'.format(seed, Ntest), test_in)
        # np.savetxt('drive_torque_seed{}_n{}.txt'.format(seed, Ntest), torques)
        # np.savetxt('time.txt', time_x)
    
    