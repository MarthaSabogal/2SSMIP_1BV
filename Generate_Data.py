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

N = 400                      # number of scenarios
#gbar = 337262 + 1953823     # nominal global production
#r = 0.65                    # threshold for geopolitical strain 
                
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

# Big M values
#M1 = r*gbar                    # r*g 
#M2 = 120000                    # capacity of plants 
#M3 = sum(q_s[i] for i in I)
#%% Stochastic Data
# to build scenario set
d, gamma_s, gamma_p, xi_qs, xi_qp, xi_nd = {}, {}, {}, {}, {}, {}
U_cal, F_cal = {}, {}                                                  # origin and propagation of natural disasters (ND)
Ubar_cal, Fbar_cal = {}, {}                                            # to model if manufacturing capacity is shut down
Kbar, K_Kbar  = {}, {}                                                 # set of countries where: a ND occurred as origin, a ND did not occur as origin

total_demand = np.zeros(N)
others_prod = np.zeros(N)

for w in range(N):  
    Kbar[w] = []                        
    K_Kbar[w] = []               
    
    for k in K:
        d[k,w] = np.ceil(np.random.normal(demand[k][0], demand[k][1]))  # demand (mean,stv) 
        if d[k,w] < 0:
            d[k,w] = 0
        total_demand[w] += d[k,w]
        
        U_cal[k,w] = np.random.binomial(1,1-psi[k])                     # origin ND (if 0, then ND occurred)
        if U_cal[k,w] == 0:
            Kbar[w].append(k)
            xi_nd[k,w] = np.random.binomial(1,1-pr_shutdown[k])         # shut down capacity?
        else:
            K_Kbar[w].append(k)
            
    for k in K_Kbar[w]:
        counter = 0
        for c in Hk[k]:
            if c in Kbar[w]:
                counter += 1                
        if counter == 0:
            xi_nd[k,w] = 1
        
        else:
            AUX = set(Hk[k]).intersection(Kbar[w])                      # propagation of ND 
            for kprime in AUX:
                F_cal[k,kprime,w] = np.random.binomial(1,1-p_nd[k,kprime])   
            if min(F_cal[k,kprime,w] for kprime in AUX) == 0:
                xi_nd[k,w] = np.random.binomial(1,1-pr_shutdown[k])               
            else: 
                xi_nd[k,w] = 1
         
    for i in I:
        xi_qs[i,w] = np.random.binomial(1, p_qs[i])                             # supplier, quality disruption
        sprobabilities = [supplier_df2.loc[i, c] for c in cap_ava]              
        gamma_s[i,w] = np.random.choice(cap_ava, 1, p=sprobabilities)[0]        # supplier, quality strains

    for j in J:
        L1_probabilities = [gammaL1_df.loc[j, c] for c in cap_ava]         
        gamma_p[j,1,w] = np.random.choice(cap_ava, 1, p=L1_probabilities)[0]    # plants, quality strains
      
        L2_probabilities = [gammaL2_df.loc[j, c] for c in cap_ava]         
        gamma_p[j,2,w] = np.random.choice(cap_ava, 1, p=L2_probabilities)[0]  
        
        L3_probabilities = [gammaL3_df.loc[j, c] for c in cap_ava]         
        gamma_p[j,3,w] = np.random.choice(cap_ava, 1, p=L3_probabilities)[0]  
  
        for l in L:
            xi_qp[j,l,w] = np.random.binomial(1,p_qp[j,l])                      # plants, quality disruption

    for k in K: 
        others_prod[w] += eta[k]*xi_nd[k,w]

#%%Store scenarios data
# Code that store scenarios data, used to run all methods with the same sample in separate .py files

dfdemand = pd.DataFrame(columns=["ISO","scenario","d"])
dfdisaster = pd.DataFrame(columns=["ISO","scenario","xi_nd"])
dfstrainsp = pd.DataFrame(columns=["ISO","quality","scenario","gamma_p"])
dfqualityp = pd.DataFrame(columns=["ISO","quality","scenario","xi_qp"])
dfstrainss = pd.DataFrame(columns=["ISO","scenario","gamma_s"])
dfqualitys = pd.DataFrame(columns=["ISO","scenario","xi_qs"])

for w in range(N):
    for k in K:
        dfdemand = pd.concat([dfdemand, pd.DataFrame.from_records([{"ISO":k,"scenario":w,"d":d[k,w]}])], ignore_index=True)
        dfdisaster = pd.concat([dfdisaster, pd.DataFrame.from_records([{"ISO":k,"scenario":w,"xi_nd":xi_nd[k,w]}])], ignore_index=True)

    for j in J:
        for l in L:
            dfstrainsp = pd.concat([dfstrainsp, pd.DataFrame.from_records([{"ISO":j,"quality":l,"scenario":w,"gamma_p":gamma_p[j,l,w]}])], ignore_index=True)
            dfqualityp = pd.concat([dfqualityp, pd.DataFrame.from_records([{"ISO":j,"quality":l,"scenario":w,"xi_qp":xi_qp[j,l,w]}])], ignore_index=True)
    
    for i in I:
        dfstrainss = pd.concat([dfstrainss, pd.DataFrame.from_records([{"ISO":i,"scenario":w,"gamma_s":gamma_s[i,w]}])], ignore_index=True)
        dfqualitys = pd.concat([dfqualitys, pd.DataFrame.from_records([{"ISO":i,"scenario":w,"xi_qs":xi_qs[i,w]}])], ignore_index=True)

dfdemand.to_csv("demand",index=False)
dfdisaster.to_csv("disaster",index=False)
dfstrainsp.to_csv("strainsp",index=False)
dfqualityp.to_csv("qualityp",index=False)
dfstrainss.to_csv("strainss",index=False)
dfqualitys.to_csv("qualitys",index=False)
