# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:16:31 2024

@author: msaboga
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time

#%% Sets
sets_df = pd.read_excel('Data.xlsx', sheet_name='sets')  

I = sets_df['I'].dropna().tolist()              # supplier countries
J = sets_df['J'].dropna().tolist()              # potential plants
K = sets_df['K'].dropna().tolist()              # demand points
L = sets_df['L'].dropna().tolist()              # quality levels
Jbar = sets_df['J_bar'].dropna().tolist()       # countries with export bans

plant_arcs = [(j,jprime) for j in J for jprime in J if jprime != j]    # arcs between plants (j=jprime is not here) 
int_arcs = [(j,k) for j in Jbar for k in K if k!=j]                    # international arcs for demand (j=k is not here) 

#%% Deterministic Data 

N = 500                      # number of scenarios
se = 505                     # seed

def nominalpn():           # from nominal model
    return 337784

def rthreshold():
    return 0.95

def Nvalue():
    return N

def seedvalue():
    return se

def objnm():
    return 1701923
 
# Common Hazards Set
Hk_df = pd.read_excel('Data.xlsx', sheet_name='H_k')     
Hk = {}
for k in K:
    Hk[k] = Hk_df[k].dropna().tolist() 

# suppliers parameters: raw material costs, capacity, rho_qua_suppliers 
supplier_df = pd.read_excel('Data.xlsx', sheet_name='I')
ISOS, c_s, q_s, p_qs = gp.multidict(supplier_df[['ISO','c_s','q_s','psq']].set_index('ISO').T.to_dict('list'))

# plants parameters: production costs, holding costs, capaciy
plant_df = pd.read_excel('Data.xlsx', sheet_name='J')
ISOP, c_p, c_h, q_p = gp.multidict(plant_df[['ISO','c_p','c_h','q_p']].set_index('ISO').T.to_dict('list'))  

# rho_qua_plants (prob. of bernoulli distributions for plants)
pqua_df = pd.read_excel('Data.xlsx', sheet_name='p_qua')
pqua_df = pqua_df.melt(id_vars=["ISO"], var_name="quality", value_name="pqua")   
pqua_df = pqua_df.set_index(['ISO','quality'])     
p_qp = dict(zip(pqua_df.index, pqua_df.pqua))  

# fixed cost of plants 
qualcosts_df = pd.read_excel('Data.xlsx', sheet_name='cfi')
qualcosts_df = qualcosts_df.melt(id_vars=["ISO"], var_name="quality", value_name="c_f")    
qualcosts_df = qualcosts_df.set_index(['ISO','quality'])    
c_f = dict(zip(qualcosts_df.index, qualcosts_df.c_f))     

# clients parameters: price, exports, pr. shutdown a plant, pr.nd occurs
client_df = pd.read_excel('Data.xlsx', sheet_name='K')
ISOD, p, eta, pr_shutdown, psi = gp.multidict(client_df[['ISO','p', 'eta','pr_shutdown','pr_nd']].set_index('ISO').T.to_dict('list'))

# probability of bernoulli distributions for propagation effects
pnd_df = pd.read_excel('Data.xlsx', sheet_name='p_kk')  
pnd_df = pnd_df.melt(id_vars=["ISO"], var_name="origin_disaster", value_name="p_nd")   
pnd_df = pnd_df.set_index(['ISO','origin_disaster'])     
p_nd = dict(zip(pnd_df.index, pnd_df.p_nd))     

# transportation costs
dft1 = pd.read_excel('Data.xlsx', sheet_name='c_u')
dft1 = dft1.melt(id_vars=["ISO"], var_name="To", value_name="c_u")  
dft1 = dft1.set_index(['ISO','To'])     
c_u = dict(zip(dft1.index, dft1.c_u))

dft2 = pd.read_excel('Data.xlsx', sheet_name='c_v')
dft2 = dft2.melt(id_vars=["ISO"], var_name="To", value_name="c_v")
dft2 = dft2.set_index(['ISO','To'])
c_v = dict(zip(dft2.index, dft2.c_v))

c_tau = {}
for (j,jprime) in plant_arcs:
    c_tau[j,jprime] = c_v[j,jprime]
    
# Distributions parameters
# demand parameters (mean and stv of normal dist)
demand = client_df[['ISO','d_m','d_stv']].set_index('ISO').T.to_dict('list') 

# probability mass function for supplier and production strains
cap_ava = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]

supplier_df2 = supplier_df[['ISO']+cap_ava].set_index('ISO')

gammaL1_df = pd.read_excel('Data.xlsx', sheet_name='L1') 
gammaL1_df = gammaL1_df[['ISO']+cap_ava].set_index('ISO')   

gammaL2_df = pd.read_excel('Data.xlsx', sheet_name='L2') 
gammaL2_df = gammaL2_df[['ISO']+cap_ava].set_index('ISO')  

gammaL3_df = pd.read_excel('Data.xlsx', sheet_name='L3') 
gammaL3_df = gammaL3_df[['ISO']+cap_ava].set_index('ISO')  

#%% Stochastic Data
# to build scenario set
def disruptions(I, J, K, N, psi, pr_shutdown, Hk, p_nd, p_qs, p_qp, se):  # Sampling, Random parameters
    np.random.seed(se)              
    xi_qs, xi_qp, xi_nd = {}, {}, {}    
    U_cal, F_cal = {}, {}                                                  # origin and propagation of natural disasters (ND)
    Kbar, K_Kbar  = {}, {}                                                 # set of countries where: a ND occurred as origin, a ND did not occur as origin
    counter = {}
    AUXSETK = {}
    AUX = {}
    
    for w in range(N): 
        Kbar[w] = []                        
        K_Kbar[w] = []               
        AUXSETK[w] = []
        
        for k in K:
            U_cal[k,w] = np.random.binomial(1,1-psi[k])                     # origin ND (if 0, then ND occurred)
            if U_cal[k,w] == 0:                                             # origin ND 
                Kbar[w].append(k)
            else:
                K_Kbar[w].append(k)                                         # No origin ND
        
        for k in Kbar[w]:
            xi_nd[k,w] = np.random.binomial(1,1-pr_shutdown[k])             # shut down capacity?
            
        for k in K_Kbar[w]:
            counter[k] = 0
            for c in Hk[k]:
                if c in Kbar[w]:
                    counter[k] += 1                
            if counter[k] == 0:
                xi_nd[k,w] = 1
            else:
                AUXSETK[w].append(k)
                
        for k in AUXSETK[w]:
            AUX[k] = sorted(set(Hk[k]).intersection(Kbar[w]))                      # propagation of ND 
            for kprime in AUX[k]:
                F_cal[k,kprime,w] = np.random.binomial(1,1-p_nd[k,kprime])
            if min(F_cal[k,kprime,w] for kprime in AUX[k]) == 0:          
                xi_nd[k,w] = np.random.binomial(1,1-pr_shutdown[k])                # shut down capacity?
            else: 
                xi_nd[k,w] = 1
        
        for i in I:
            xi_qs[i,w] = np.random.binomial(1, p_qs[i])                             # supplier, quality disruption
    
        for j in J:
            for l in L:
                xi_qp[j,l,w] = np.random.binomial(1,p_qp[j,l])                      # plants, quality disruption   
            
    return xi_qs, xi_qp, xi_nd 

def oper_strains(I, J, K, N, demand, supplier_df2, cap_ava, gammaL1_df, gammaL2_df, gammaL3_df, se):  # Sampling, Random parameters
    np.random.seed(se)              
    d, gamma_s, gamma_p = {}, {}, {}

    for w in range(N):  
        for k in K:
            d[k,w] = np.ceil(np.random.normal(demand[k][0], demand[k][1]))  # demand (mean,stv) 
            if d[k,w] < 0:
                d[k,w] = 0
             
        for i in I:
            sprobabilities = [supplier_df2.loc[i, c] for c in cap_ava]              
            gamma_s[i,w] = np.random.choice(cap_ava, 1, p=sprobabilities)[0]        # supplier, quality strains
    
        for j in J:
            L1_probabilities = [gammaL1_df.loc[j, c] for c in cap_ava]         
            gamma_p[j,1,w] = np.random.choice(cap_ava, 1, p=L1_probabilities)[0]    # plants, quality strains
          
            L2_probabilities = [gammaL2_df.loc[j, c] for c in cap_ava]         
            gamma_p[j,2,w] = np.random.choice(cap_ava, 1, p=L2_probabilities)[0]  
            
            L3_probabilities = [gammaL3_df.loc[j, c] for c in cap_ava]         
            gamma_p[j,3,w] = np.random.choice(cap_ava, 1, p=L3_probabilities)[0]  
            
    return d, gamma_s, gamma_p