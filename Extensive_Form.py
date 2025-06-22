# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:16:31 2024

@author: Martha Sabogal
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time
import Generate_Data

#%% Sets
sets_df = pd.read_excel('Data.xlsx', sheet_name='sets')  

I = sets_df['I'].dropna().tolist()              # supplier countries
J = sets_df['J'].dropna().tolist()              # potential plants
K = sets_df['K'].dropna().tolist()              # demand points
L = sets_df['L'].dropna().tolist()              # quality levels
Jbar = sets_df['J_bar'].dropna().tolist()       # countries with export bans


plant_arcs = [(j,jprime) for j in J for jprime in J if jprime != j]    # arcs between plants (j=jprime is not here) 
int_arcs = [(j,k) for j in Jbar for k in K if k!=j]                    # international arcs affected by export bans (j=k is not here) 

#%% Deterministic Data 
N = Generate_Data.Nvalue()                           # number of scenarios
nv = Generate_Data.nominalpn()                       # nominal company production  
gbar = nv + 1953823                                  # nominal global production  
r = Generate_Data.rthreshold()                       # threshold for geopolitical strain 
se = Generate_Data.seedvalue()                       # seed     

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
M1 = r*gbar                     # r*g 
M2 = 120000                     # capacity of plants 
M3 = sum(q_s[i] for i in I)     # total suppliers capacity
#%% Stochastic Data 
d, gamma_s, gamma_p, xi_qs, xi_qp, xi_nd = {}, {}, {}, {}, {}, {}
d, gamma_s, gamma_p = Generate_Data.oper_strains(I, J, K, N, demand, supplier_df2, cap_ava, gammaL1_df, gammaL2_df, gammaL3_df, se)
xi_qs, xi_qp, xi_nd = Generate_Data.disruptions(I, J, K, N, psi, pr_shutdown, Hk, p_nd, p_qs, p_qp, se) 
             
total_demand = np.zeros(N)
others_prod = np.zeros(N)
for w in range(N):
    for k in K:
        total_demand[w] += d[k,w]
        others_prod[w] += eta[k]*xi_nd[k,w]
        
filename = f"gurobi_stats_EF_II_{se}_{int(r*100)}.txt"
print(filename)

#%% Extensive Form
ini_time_EF = time.time()

m_ef = gp.Model("Extensive_Form")

# Variables
y_ef, z_ef = {}, {}

for j in J:
    for l in L:
        y_ef[j,l] = m_ef.addVar(vtype=GRB.BINARY)

for w in range(N):
    z_ef[w] = m_ef.addVar(vtype=GRB.BINARY)

u_ef, x_ef, v_ef, e_ef, tau_ef = {}, {}, {}, {}, {}

for w in range(N):
    for j in J:
        x_ef[j,w] = m_ef.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
        e_ef[j,w] = m_ef.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
        for i in I:
            u_ef[i,j,w] = m_ef.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
        for k in K:
            v_ef[j,k,w] = m_ef.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
    for (j,jprime) in plant_arcs:
        tau_ef[j,jprime,w] = m_ef.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
              
m_ef.update()

# Objective Function
m_ef.setObjective(gp.quicksum(gp.quicksum(-c_f[j,l]*y_ef[j,l] for l in L) for j in J)+
               1/N*gp.quicksum(gp.quicksum(gp.quicksum((p[k]-c_p[j]-c_v[j,k])*v_ef[j,k,w] for k in K) for j in J) 
                    - gp.quicksum(gp.quicksum((c_s[i]+c_u[i,j])*u_ef[i,j,w] for j in J) for i in I)
                                 - gp.quicksum(c_h[j]*e_ef[j,w] for j in J)
                                 - gp.quicksum(c_tau[j,jprime]*tau_ef[j,jprime,w] for (j,jprime) in plant_arcs) for w in range(N)), gp.GRB.MAXIMIZE)

# Constraints
c1_1, c2_1, c2_2, c2_3, c2_4, c2_5, c2_6, c2_7, c2_8, c2_9, c2_10 = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

for j in J:
    c1_1[j] = m_ef.addConstr(gp.quicksum(y_ef[j,l] for l in L) <= 1)
        
for w in range(N):
    for i in I:
        c2_1[i,w] = m_ef.addConstr(gp.quicksum(u_ef[i,j,w] for j in J) <= q_s[i]*gamma_s[i,w]*xi_nd[i,w]*xi_qs[i,w])
   
    for j in J:
        c2_2[j,w] = m_ef.addConstr(x_ef[j,w] <= q_p[j]*xi_nd[j,w]*gp.quicksum(gamma_p[j,l,w]*xi_qp[j,l,w]*y_ef[j,l] for l in L))
        c2_3[j,w] = m_ef.addConstr(gp.quicksum(u_ef[i,j,w] for i in I) - x_ef[j,w] == 0)
        c2_5[j,w] = m_ef.addConstr(gp.quicksum(v_ef[j,k,w] for k in K) <= q_p[j]*xi_nd[j,w]*gp.quicksum(gamma_p[j,l,w]*xi_qp[j,l,w]*y_ef[j,l] for l in L))
        c2_6[j,w] = m_ef.addConstr(gp.quicksum(v_ef[j,k,w] for k in K) + e_ef[j,w] + gp.quicksum(tau_ef[j,jprime,w] for jprime in J if jprime != j) == 
                                   gp.quicksum(u_ef[i,j,w] for i in I) + gp.quicksum(tau_ef[jprime,j,w] for jprime in J if jprime != j))
        c2_7[j,w] = m_ef.addConstr(gp.quicksum(tau_ef[jprime,j,w] for jprime in J if jprime != j) <= M3*gp.quicksum(y_ef[j,l] for l in L))

    c2_4[w] = m_ef.addConstr(gp.quicksum(x_ef[j,w] for j in J) <= gp.quicksum(d[k,w] for k in K))
          
    for k in K:
        c2_8[k,w] = m_ef.addConstr(gp.quicksum(v_ef[j,k,w] for j in J) <= d[k,w])    
    
    c2_9[w] = m_ef.addConstr(r*gbar - gp.quicksum(x_ef[j,w] for j in J) - gp.quicksum(eta[k]*xi_nd[k,w] for k in K) <= (1-z_ef[w])*M1)
    
    for (j,k) in int_arcs:
        c2_10[j,k,w] = m_ef.addConstr(v_ef[j,k,w] <= z_ef[w]*M2)

m_ef.update()
m_ef.setParam("OutputFlag", 1)      
m_ef.setParam('TimeLimit',3600)
m_ef.setParam("Threads", 1)

ini_time_OEF = time.time()
m_ef.optimize()
timeOEF = time.time()-ini_time_OEF        # Optimization time
timeEF = time.time()-ini_time_EF          # Optimization + setup of models

#%%Basic Results
print('Opt+Setup Time EF', timeEF)
print('Optimization Time EF', timeOEF)    

ZEF = m_ef.objVal
y_efval = {}
for j in J:
    for l in L:
        y_efval[j,l] = y_ef[j,l].x
        if y_efval[j,l] > 0.9:
            print("{}{}:".format(j,l),y_efval[j,l])

Fixed_costs = sum(y_efval[j,l]*c_f[j,l] for j in J for l in L)
Second_Stage_OF = ZEF + Fixed_costs

nodes_explored = m_ef.NodeCount                                 
gap = m_ef.MIPGap                                               

# Save the statistics and results
with open(filename, "w") as file:
    file.write("Scenarios,Method,Nodes,Opt Time,Opt+Setup Time,Gap,Master Obj,Fixed Costs,2-Stage OF,r,seed,status\n")
    file.write(f"{N},EF_II,{nodes_explored},{timeOEF},{timeEF},{gap},{ZEF},{Fixed_costs},{Second_Stage_OF},{r},{se},{m_ef.status}\n")
    
print("Objective:", ZEF)
print("Fixed Cost:", Fixed_costs) 
print("2-Stage OF:", Second_Stage_OF)

#%%Results 
if m_ef.status == GRB.OPTIMAL:
    mean_tau = {(j,jprime): 0 for (j,jprime) in plant_arcs}
    mean_e = {j: 0 for j in J}
    mean_x = {j: 0 for j in J}  
    mean_v = {(j,k): 0 for j in J for k in K}
    mean_u = {(i,j): 0 for i in I for j in J}
    z_efval = np.zeros(N)
    
    tau_efval, e_efval, x_efval, v_efval, u_efval = {}, {}, {}, {}, {}
    
    for w in range(N):
        z_efval[w] = z_ef[w].x
        
        for (j,jprime) in plant_arcs:
            mean_tau[j,jprime] += (1/N)*tau_ef[j,jprime,w].x   
            #tau_efval[j,jprime,w] = tau_ef[j,jprime,w].x        
        
        for j in J:
            mean_e[j] += (1/N)*e_ef[j,w].x
            #e_efval[j,w] = e_ef[j,w].x
            
            mean_x[j] += (1/N)*x_ef[j,w].x  
            x_efval[j,w] = x_ef[j,w].x
        
            for k in K:
                mean_v[j,k] += (1/N)*v_ef[j,k,w].x
                #v_efval[j,k,w] = v_ef[j,k,w].x
                
            for i in I:    
                mean_u[i,j] += (1/N)*u_ef[i,j,w].x
                #u_efval[i,j,w] = u_ef[i,j,w].x
 
    print("total v:", sum(mean_v[j,k] for j in J for k in K))
    print("total x:", sum(mean_x[j] for j in J))
    
    global_pn = np.zeros(N)
    z_real = np.zeros(N)
    for w in range(N):
        global_pn[w] = others_prod[w]+sum(x_efval[j,w] for j in J)
        if global_pn[w] < r*gbar:
            print("z must be 0, short drug supply in scenario:", w)
            z_real[w] = 0
        else:
            z_real[w] = 1            
        
    # Data frames Means
    tau_df = pd.DataFrame(columns=["From","To","Tau"])
    x_df = pd.DataFrame(columns=["ISO","x","e"])
    v_df = pd.DataFrame(columns=["From","To","v"])
    u_df = pd.DataFrame(columns=["From","To","u"])
    
    for (j,jprime) in plant_arcs:
        if mean_tau[j,jprime] > 0.1:
            tau_df = pd.concat([tau_df, pd.DataFrame.from_records([{"From":j,"To":jprime,"Tau":mean_tau[j,jprime]}])], ignore_index=True)       
    
    for j in J:
        x_df = pd.concat([x_df, pd.DataFrame.from_records([{"ISO":j,"x":mean_x[j],"e":mean_e[j]}])], ignore_index=True)   
    
        for k in K:
            if mean_v[j,k] > 0.1:
                v_df = pd.concat([v_df, pd.DataFrame.from_records([{"From":j,"To":k, "v":mean_v[j,k]}])], ignore_index=True)
    
        for i in I:
            if mean_u[i,j] > 0.1:
                u_df = pd.concat([u_df, pd.DataFrame.from_records([{"From":i,"To":j, "u":mean_u[i,j]}])], ignore_index=True)
    
    # Data frames Scenarios
    z_scenarios_df = pd.DataFrame(columns=["Scenario", "z"])
    globalpn_scenarios_df = pd.DataFrame(columns=["Scenario", "gp","actual z"])
     
    for w in range(N):
        z_scenarios_df = pd.concat([z_scenarios_df, pd.DataFrame.from_records([{"Scenario":w,"z":z_efval[w]}])], ignore_index=True)
        globalpn_scenarios_df = pd.concat([globalpn_scenarios_df, pd.DataFrame.from_records([{"Scenario":w,"gp":global_pn[w], "actual z":z_real[w]}])], ignore_index=True) 
    
    filename2 = f"Results_Disruption_Model_II_{se}_{int(r*100)}.xlsx"
    # Exporting results
    with pd.ExcelWriter(filename2) as writer:
        v_df.to_excel(writer,"v")  
        u_df.to_excel(writer,"u")
        x_df.to_excel(writer,"x-e")
        tau_df.to_excel(writer,"tau")
        z_scenarios_df.to_excel(writer,"z-w")
        globalpn_scenarios_df.to_excel(writer,"gp-zreal")
        