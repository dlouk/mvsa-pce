# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:13:24 2023

@author: z004dp6c
"""

import openturns as ot

# parameter distributions --> a bit tricky for Lognormal    
b_para = ot.LogNormalMuSigma(0.15, 0.0075)
dist_b = ot.ParametrizedDistribution(b_para)
#
w_para = ot.LogNormalMuSigma(0.3, 0.015)
dist_w = ot.ParametrizedDistribution(w_para)
#
L_para = ot.LogNormalMuSigma(5.0, 0.05)
dist_L = ot.ParametrizedDistribution(L_para)
#
E_para = ot.LogNormalMuSigma(3e10, 4.5e9) 
dist_E = ot.ParametrizedDistribution(E_para)
#
P_para = ot.LogNormalMuSigma(1e4, 2e3) 
dist_P = ot.ParametrizedDistribution(P_para)
#
# add dummy distributions
dummy_para = ot.LogNormalMuSigma(10, 1) 
dummy1 = ot.ParametrizedDistribution(dummy_para)
dummy2 = ot.ParametrizedDistribution(dummy_para)
dummy3 = ot.ParametrizedDistribution(dummy_para)
dummy4 = ot.ParametrizedDistribution(dummy_para)
dummy5 = ot.ParametrizedDistribution(dummy_para)
dummy6 = ot.ParametrizedDistribution(dummy_para)
dummy7 = ot.ParametrizedDistribution(dummy_para)
dummy8 = ot.ParametrizedDistribution(dummy_para)
dummy9 = ot.ParametrizedDistribution(dummy_para)
dummy10 = ot.ParametrizedDistribution(dummy_para)
dummy11 = ot.ParametrizedDistribution(dummy_para)
dummy12 = ot.ParametrizedDistribution(dummy_para)
dummy13 = ot.ParametrizedDistribution(dummy_para)
dummy14 = ot.ParametrizedDistribution(dummy_para)
dummy15 = ot.ParametrizedDistribution(dummy_para)
# 
# joint distribution
dist_joint = ot.ComposedDistribution([dist_b, dist_w, dist_L, dist_E, dist_P, 
                                      dummy1, dummy2, dummy3, dummy4, dummy5, 
                                      dummy6, dummy7, dummy8, dummy9, dummy10, 
                                      dummy11, dummy12, dummy13, dummy14, 
                                      dummy15])

# convert to normal distributions
b_muLog = b_para.evaluate()[0]
b_sigmaLog = b_para.evaluate()[1]
dist_b_normal = ot.Normal(b_muLog, b_sigmaLog)
#
w_muLog = w_para.evaluate()[0]
w_sigmaLog = w_para.evaluate()[1]
dist_w_normal = ot.Normal(w_muLog, w_sigmaLog)
#
L_muLog = L_para.evaluate()[0]
L_sigmaLog = L_para.evaluate()[1]
dist_L_normal = ot.Normal(L_muLog, L_sigmaLog)
#
E_muLog = E_para.evaluate()[0]
E_sigmaLog = E_para.evaluate()[1]
dist_E_normal = ot.Normal(E_muLog, E_sigmaLog)
#
P_muLog = P_para.evaluate()[0]
P_sigmaLog = P_para.evaluate()[1]
dist_P_normal = ot.Normal(P_muLog, P_sigmaLog)
# 
dummy_muLog = dummy_para.evaluate()[0]
dummy_sigmaLog = dummy_para.evaluate()[1]
dummy1_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy2_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy3_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy4_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy5_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy6_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy7_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy8_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy9_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy10_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy11_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy12_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy13_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy14_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
dummy15_normal = ot.Normal(dummy_muLog, dummy_sigmaLog)
# 
dist_joint_normal = ot.ComposedDistribution([dist_b_normal, dist_w_normal, 
                                             dist_L_normal, dist_E_normal, 
                                             dist_P_normal, dummy1_normal, 
                                             dummy2_normal, dummy3_normal, 
                                             dummy4_normal, dummy5_normal, 
                                             dummy6_normal, dummy7_normal, 
                                             dummy8_normal, dummy9_normal, 
                                             dummy10_normal, dummy11_normal, 
                                             dummy12_normal, dummy13_normal, 
                                             dummy14_normal, dummy15_normal])



