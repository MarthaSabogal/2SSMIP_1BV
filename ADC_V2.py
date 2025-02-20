# -*- coding: utf-8 -*-
"""
Created on Tue Feb  14 13:58:52 2025

@author: Martha Sabogal
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time
from sys import exit

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

N = 50                      # number of scenarios
gbar = 337262 + 1953823     # nominal global production
r = 0.95                    # threshold for geopolitical strain

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
#%%Read scenarios from stored data 

d, gamma_s, gamma_p, xi_qs, xi_qp, xi_nd = {}, {}, {}, {}, {}, {}

demand = pd.read_csv('demand.csv')
d_df = demand.set_index(['ISO','scenario'])    
d = dict(zip(d_df.index, d_df.d)) 
 
disaster = pd.read_csv('disaster.csv')
xidis_df = disaster.set_index(['ISO','scenario'])    
xi_nd = dict(zip(xidis_df.index, xidis_df.xi_nd)) 
 
strainsp = pd.read_csv('strainsp.csv')
gammap_df = strainsp.set_index(['ISO','quality','scenario'])    
gamma_p = dict(zip(gammap_df.index, gammap_df.gamma_p))  
 
strainss = pd.read_csv('strainss.csv')
gammas_df = strainss.set_index(['ISO','scenario'])    
gamma_s = dict(zip(gammas_df.index, gammas_df.gamma_s))

qualitys = pd.read_csv('qualitys.csv')
xisq_df = qualitys.set_index(['ISO','scenario'])   
xi_qs = dict(zip(xisq_df.index, xisq_df.xi_qs))
 
qualityp = pd.read_csv('qualityp.csv')
xiqua_df = qualityp.set_index(['ISO','quality','scenario'])    
xi_qp = dict(zip(xiqua_df.index, xiqua_df.xi_qp))  

total_demand = np.zeros(N)
others_prod = np.zeros(N)
for w in range(N):
    for k in K:
        total_demand[w] += d[k,w]
        others_prod[w] += eta[k]*xi_nd[k,w]
        
min_cp =  min(c_p[j] for j in J)
min_cs =  min(c_s[i] for i in I)    
U = sum(d[k,w]*(p[k]-min_cp-min_cs) for k in K for w in range(N))*1/N   #upper bound for second stage obj. function
#M = 1689827                                                            #upper bound for Obj Master from nominal model

#%% Subproblem LP
ini_time_ADC = time.time()

m_subp = gp.Model('Subproblem')

# Decision Variables Subproblem
u, x, v, e, tau = {}, {}, {}, {}, {}

z = m_subp.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)

for j in J:
    x[j] = m_subp.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
    e[j] = m_subp.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
    for i in I:
        u[i,j] = m_subp.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
    for k in K:
        v[j,k] = m_subp.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)

for (j,jprime) in plant_arcs:
    tau[j,jprime] = m_subp.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)

# Objective function  
m_subp.setObjective(gp.quicksum(gp.quicksum((p[k]-c_p[j]-c_v[j,k])*v[j,k] for k in K) for j in J) 
            - gp.quicksum(gp.quicksum((c_s[i]+c_u[i,j])*u[i,j] for j in J) for i in I)
                         - gp.quicksum(c_h[j]*e[j] for j in J)
                         - gp.quicksum(c_tau[j,jprime]*tau[j,jprime] for (j,jprime) in plant_arcs), gp.GRB.MAXIMIZE)

# Subproblem constraints
cs2_1, cs2_2, cs2_3, cs2_4, cs2_5, cs2_6, cs2_7, cs2_8, cs2_9, cs2_10 = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

for i in I:
    cs2_1[i] = m_subp.addConstr(gp.quicksum(u[i,j] for j in J) <= q_s[i]*gamma_s[i,0]*xi_nd[i,0]*xi_qs[i,0])
   
for j in J:
    cs2_2[j] = m_subp.addConstr(x[j] <= q_p[j]*xi_nd[j,0]*gp.quicksum(gamma_p[j,l,0]*xi_qp[j,l,0] for l in L))
    cs2_3[j] = m_subp.addConstr(gp.quicksum(u[i,j] for i in I) - x[j] == 0)
    cs2_5[j] = m_subp.addConstr(gp.quicksum(v[j,k] for k in K) <= q_p[j]*xi_nd[j,0]*gp.quicksum(gamma_p[j,l,0]*xi_qp[j,l,0] for l in L))
    cs2_6[j] = m_subp.addConstr(gp.quicksum(v[j,k] for k in K) + e[j] + gp.quicksum(tau[j,jprime] for jprime in J if jprime != j) 
                                - gp.quicksum(u[i,j] for i in I) - gp.quicksum(tau[jprime,j] for jprime in J if jprime != j) == 0)
    cs2_7[j] = m_subp.addConstr(gp.quicksum(tau[jprime,j] for jprime in J if jprime != j) <= M3 )

cs2_4 = m_subp.addConstr(gp.quicksum(x[j] for j in J) <= gp.quicksum(d[k,0] for k in K))

for k in K:
    cs2_8[k] = m_subp.addConstr(gp.quicksum(v[j,k] for j in J) <= d[k,0])   

cs2_9 = m_subp.addConstr(M1*z - gp.quicksum(x[j] for j in J) <= gp.quicksum(eta[k]*xi_nd[k,0] for k in K))

for (j,k) in int_arcs:
    cs2_10[j,k] = m_subp.addConstr(v[j,k] - M2*z <= 0)
    
m_subp.update()

#%% CGLP

m_disj = gp.Model('CGLP')

# Set and initialize paramaters
H = {0 , 1}                                  # Union of polyhedra z=0 and z=1
yval, t_alpha0, t_alpha1 = {}, {}, {}
for j in J:
    for l in L:
        yval[j,l] = 0
        t_alpha0[j,l] = 0
        t_alpha1[j,l] = 0
t_beta0 = 0
t_beta1 = 0

# Decision variables
delta_bar = m_disj.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=1)
beta_bar = m_disj.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=-1)

alpha_bar = {}
lambda_c, lambda_m, lambda_u = {},{},{}

for j in J:
    for l in L:
        alpha_bar[j,l] = m_disj.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=yval[j,l])

for h in H:
    lambda_c[h] = m_disj.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, obj=0)
    for j in J:
        lambda_m[h,j] = m_disj.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, obj=0)
        for l in L:
            lambda_u[h,j,l] = m_disj.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, obj=0)

# Objective function     
m_disj.ModelSense = GRB.MAXIMIZE

# Constraints
cd2_0, cd2_1, cd3 = {}, {}, {}

cd1_0 = m_disj.addConstr(t_beta0*lambda_c[0] + gp.quicksum(1*lambda_m[0,j]+gp.quicksum(1*lambda_u[0,j,l] for l in L) for j in J)-beta_bar <= 0) 
cd1_1 = m_disj.addConstr(t_beta1*lambda_c[1] + gp.quicksum(1*lambda_m[1,j]+gp.quicksum(1*lambda_u[1,j,l] for l in L) for j in J)-beta_bar <= 0) 

for j in J:
    for l in L:
        cd2_0[j,l] = m_disj.addConstr(alpha_bar[j,l]+t_alpha0[j,l]*lambda_c[0]-1*lambda_m[0,j]-1*lambda_u[0,j,l] <= 0)
        cd2_1[j,l] = m_disj.addConstr(alpha_bar[j,l]+t_alpha1[j,l]*lambda_c[1]-1*lambda_m[1,j]-1*lambda_u[1,j,l] <= 0)
        
for h in H:
    cd3[h] = m_disj.addConstr(delta_bar - lambda_c[h] <= 0)

cd4 = m_disj.addConstr(gp.quicksum(lambda_c[h] for h in H) == 1) 

#%% functions

# To create master problem
def create_mp():
    m_mp = gp.Model('Master_Single')
    
    # Decision variables
    m_mp._y = {}
    for j in J:
        for l in L:
            m_mp._y[j,l] = m_mp.addVar(vtype=GRB.BINARY, obj=-c_f[j,l])
    m_mp._theta = m_mp.addVar(vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = U, obj=1)
    
    # Objective Function 
    m_mp.ModelSense = GRB.MAXIMIZE
    
    # Constraints
    cm1_1 = {}
    for j in J:
        cm1_1[j] = m_mp.addConstr(gp.quicksum(m_mp._y[j,l] for l in L) <= 1)
  
    m_mp.update()
    return m_mp

# To update Subproblem RHS
def update_rhs(d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, yval, w):
    for i in I:
        cs2_1[i].setAttr(GRB.Attr.RHS, q_s[i]*gamma_s[i,w]*xi_nd[i,w]*xi_qs[i,w])
    for j in J:
        cs2_2[j].setAttr(GRB.Attr.RHS, q_p[j]*xi_nd[j,w]*sum(gamma_p[j,l,w]*xi_qp[j,l,w]*yval[j,l] for l in L))
        cs2_5[j].setAttr(GRB.Attr.RHS, q_p[j]*xi_nd[j,w]*sum(gamma_p[j,l,w]*xi_qp[j,l,w]*yval[j,l] for l in L))
        cs2_7[j].setAttr(GRB.Attr.RHS, M3*sum(yval[j,l] for l in L))
    cs2_4.setAttr(GRB.Attr.RHS, sum(d[k,w] for k in K))    
    for k in K:
        cs2_8[k].setAttr(GRB.Attr.RHS, d[k,w])    
    cs2_9.setAttr(GRB.Attr.RHS, sum(eta[k]*xi_nd[k,w] for k in K))
    m_subp.update()

# Create and initialize dual variables
dualcs2_1, dualcs2_2, dualcs2_3, dualcs2_4, dualcs2_5 = {}, {}, {}, {}, {}
dualcs2_6, dualcs2_7, dualcs2_8, dualcs2_9, dualcs2_10  = {}, {}, {}, {}, {}
for i in I:
    dualcs2_1[i] = 0   
for j in J:
    dualcs2_2[j] = 0
    dualcs2_3[j] = 0
    dualcs2_5[j] = 0
    dualcs2_6[j] = 0 
    dualcs2_7[j] = 0 
dualcs2_4 = 0
dualcs2_9 = 0
for k in K:
    dualcs2_8[k] = 0   
for (j,k) in int_arcs:
    dualcs2_10[j,k] = 0   

# Capture dual variables
def get_duals():
    for i in I:
        dualcs2_1[i] = cs2_1[i].pi  
    for j in J:
        dualcs2_2[j] = cs2_2[j].pi
        dualcs2_3[j] = cs2_3[j].pi
        dualcs2_5[j] = cs2_5[j].pi 
        dualcs2_6[j] = cs2_6[j].pi 
        dualcs2_7[j] = cs2_7[j].pi  
    dualcs2_4 = cs2_4.pi  
    dualcs2_9 = cs2_9.pi
    for k in K:
        dualcs2_8[k] = cs2_8[k].pi 
    for (j,k) in int_arcs:
        dualcs2_10[j,k] = cs2_10[j,k].pi    
        
    return dualcs2_1, dualcs2_2, dualcs2_3, dualcs2_4, dualcs2_5, dualcs2_6, dualcs2_7, dualcs2_8, dualcs2_9, dualcs2_10    

# Benders optimality cut coefficients
def coeff_benders(dualcs2_1, dualcs2_2, dualcs2_3, dualcs2_4, dualcs2_5, dualcs2_6, dualcs2_7, dualcs2_8, dualcs2_9, dualcs2_10, 
                    d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, w):
    temp_alpha = {}
    for j in J:
       for l in L:
           temp_alpha[j,l] = dualcs2_2[j]*(q_p[j]*xi_nd[j,w]*gamma_p[j,l,w]*xi_qp[j,l,w])+\
               dualcs2_5[j]*(q_p[j]*xi_nd[j,w]*gamma_p[j,l,w]*xi_qp[j,l,w])+\
                   dualcs2_7[j]*M3
           
    temp_beta = sum(dualcs2_1[i]*(q_s[i]*gamma_s[i,w]*xi_nd[i,w]*xi_qs[i,w]) for i in I)+ dualcs2_4*(sum(d[k,w] for k in K))+\
        sum(dualcs2_8[k]*d[k,w] for k in K)+dualcs2_9*sum(eta[k]*xi_nd[k,w] for k in K)
                                    
    return temp_alpha, temp_beta    

# update obj coeff. cglp
def update_obj_coe_cglp(yval):
    for j in J:
        for l in L:
            alpha_bar[j,l].Obj = yval[j,l]

    m_disj.update()

# update constraints coeff. cglp
def update_constraints_coe_cglp(t_beta0, t_beta1, t_alpha0, t_alpha1):
    m_disj.chgCoeff(cd1_0, lambda_c[0], t_beta0)
    m_disj.chgCoeff(cd1_1, lambda_c[1], t_beta1)
    
    for j in J:
        for l in L:
            m_disj.chgCoeff(cd2_0[j,l], lambda_c[0], t_alpha0[j,l])
            m_disj.chgCoeff(cd2_1[j,l], lambda_c[1], t_alpha1[j,l])
    
    m_disj.update()

# to calculate the slack of added cuts
def slack_check(model, sol):
    tight = False
    alpha, beta = None, None
    for _, (beta, alpha) in enumerate(model._cut_coef_list):
        rhs =  (1/N) * sum(beta[w] + sum(sum(alpha[j,l,w]*sol['y'][j,l] for j in J) for l in L) for w in range(N))
        lhs = sol['theta']
        if (lhs - rhs) <= 1e-5:
            tight = True
            break
    return tight, alpha, beta

#%% Algorithm
# For callback function

alpha_Qlp, beta_Qlp = {}, {}                                             # to store cut coeff.
final_alpha, final_beta, subp_obj, theta_scenarioval = {}, {}, {}, {}    # per scenario information
alpha_Qlp1 = {}                                                          # to modify dictionary

# Algorithm
def callback(model, where):
    if where == gp.GRB.Callback.MIPSOL:                             # when find an integer solution
        yval = {}
        for j in J:                                                 # take variable solutions
            for l in L:
                yval[j,l] = model.cbGetSolution(model._y[j,l])
        thetaval = model.cbGetSolution(model._theta)  
        
        # Solve Subproblem as LP -------------------------------------------------------------------------------------------------------------------
        Qlp = 0 
        ListQlpInteger0 = []     # to store scenarios where z=0
        ListQlpInteger1 = []     # to store scenarios where z=1
        QlpInteger = {}
        
        for w in range(N):
            update_rhs(d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, yval, w)
            m_subp.optimize()                   
            if m_subp.status != GRB.OPTIMAL:
                print("error! subproblem is not solved to optimality in scenario: ", w)
                exit()
            Qlp += (1/N)*m_subp.objVal
            get_duals()
            alpha_Qlp[w], beta_Qlp[w] = coeff_benders(dualcs2_1, dualcs2_2, dualcs2_3, dualcs2_4, dualcs2_5, dualcs2_6, dualcs2_7, dualcs2_8,
                                              dualcs2_9, dualcs2_10, d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, w)
            
            if z.x <= 0 + 1e-5:                      # Identify the scenarios where z was 0
                ListQlpInteger0.append(w)
                QlpInteger[w] = m_subp.objVal
                
            if z.x >= 1 - 1e-5:                      # Identify the scenarios where z was 1
                ListQlpInteger1.append(w)
                QlpInteger[w] = m_subp.objVal
            
        if (thetaval - Qlp)/max(Qlp, 1e-5) > 1e-5:
            model.cbLazy(model._theta <= (1/N)*sum(beta_Qlp[w]+sum(sum(alpha_Qlp[w][j,l]*model._y[j,l] for j in J) for l in L)  # Benders cut
                                                           for w in range(N)))
            model.update()
            alpha_Qlp1 = {(j,l,w): value
                          for w, inner_dict in alpha_Qlp.items() 
                          for (j,l), value in inner_dict.items()  }   # To change dictionary structure
            model._cut_coef_list.append((beta_Qlp, alpha_Qlp1))       # Add cut coeff. to list of added cuts of MP
        
        
        # Disjuntive cut procedure and calculate Qip -------------------------------------------------------------------------------------------------
        else: 
            update_obj_coe_cglp(yval)                                               # update cglp obj function with y solution 
            sol = {'y': yval, 'theta': thetaval}                                    # take solutions
            tight_cut, alpha_tight, beta_tight = slack_check(model=model, sol=sol)  # find the cut that is tight

            Qip = 0
            SetQlpInteger0 = set(ListQlpInteger0)
            SetQlpInteger1 = set(ListQlpInteger1)
            
            for w in range(N):
                update_rhs(d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, yval, w)
                
                # Subproblem LP was z = 0 -----------------------------------------------------------------------------------------------------
                if w in SetQlpInteger0:
                    obj_spz0 = QlpInteger[w] 
                    t_alpha0 = alpha_Qlp[w] 
                    t_beta0 = beta_Qlp[w]
                    
                    # Solve Subproblem LP z = 1
                    cs2_11 = m_subp.addConstr(-z <= -1)
                    m_subp.update()
                    m_subp.optimize()
                
                    if m_subp.status == GRB.OPTIMAL:      
                        obj_spz1 = m_subp.objVal
                        get_duals()          
                        dualcs2_11 = cs2_11.pi
                        t_alpha1, t_beta1 = coeff_benders(dualcs2_1, dualcs2_2, dualcs2_3, dualcs2_4, dualcs2_5, dualcs2_6, dualcs2_7, dualcs2_8, 
                                                dualcs2_9, dualcs2_10, d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, w) 
                        t_beta1 = t_beta1 + dualcs2_11*(-1)
                        subp_obj[w] = max(obj_spz0,obj_spz1)
                        Qip += subp_obj[w]*(1/N)
                        
                        # generate a valid cut for the union -----------------------------------------------------------------------------------
                        if tight_cut is False:
                            theta_scenarioval[w] = thetaval
                        else:
                            theta_scenarioval[w] = beta_tight[w] + sum(alpha_tight[j,l,w]*yval[j,l] for j in J for l in L)   
                        delta_bar.Obj = theta_scenarioval[w]
                        update_constraints_coe_cglp(t_beta0, t_beta1, t_alpha0, t_alpha1)
                        m_disj.optimize()
                        if delta_bar.x == 0:
                            print("error! separation problem gave a zero multiplier delta_bar for theta")
                            exit()
                        
                        final_beta[w] = beta_bar.x/delta_bar.x
                        for j in J:
                            for l in L:
                                final_alpha[j,l,w] = -alpha_bar[j,l].x/delta_bar.x         
                    else:
                        subp_obj[w] = obj_spz0
                        Qip += subp_obj[w]*(1/N)
                        final_beta[w] = t_beta0
                        for j in J:
                            for l in L:
                                final_alpha[j,l,w] = t_alpha0[j,l]
                                
                    m_subp.remove(cs2_11)
                    
                # Subproblem LP was z = 1 -----------------------------------------------------------------------------------------------------
                elif w in SetQlpInteger1:
                    obj_spz1 = QlpInteger[w] 
                    t_alpha1 = alpha_Qlp[w] 
                    t_beta1 = beta_Qlp[w]
                    
                    # Solve Subproblem LP z = 0
                    cs2_11 = m_subp.addConstr(z <= 0)
                    m_subp.update()
                    m_subp.optimize()
    
                    if m_subp.status != GRB.OPTIMAL:
                        print('error! subproblem z = 0 not solved to optimality',m_subp.status)
                        exit()
        
                    obj_spz0 = m_subp.objVal      
                    get_duals()
                    #dualcs2_11 = cs2_11.pi 
                    t_alpha0, t_beta0 = coeff_benders(dualcs2_1, dualcs2_2, dualcs2_3, dualcs2_4, dualcs2_5, dualcs2_6, dualcs2_7, dualcs2_8, 
                                                 dualcs2_9, dualcs2_10, d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, w)
                    #t_beta0 = t_beta0 + dualcs2_11*(0)
                    subp_obj[w] = max(obj_spz0,obj_spz1)
                    Qip += subp_obj[w]*(1/N)
                
                    # generate a valid cut for the union -----------------------------------------------------------------------------------
                    if tight_cut is False:
                        theta_scenarioval[w] = thetaval
                    else:
                        theta_scenarioval[w] = beta_tight[w] + sum(alpha_tight[j,l,w]*yval[j,l] for j in J for l in L)   
                    delta_bar.Obj = theta_scenarioval[w]
                    update_constraints_coe_cglp(t_beta0, t_beta1, t_alpha0, t_alpha1)
                    m_disj.optimize()
                    if delta_bar.x == 0:
                        print("error! separation problem gave a zero multiplier delta_bar for theta")
                        exit()
                    
                    final_beta[w] = beta_bar.x/delta_bar.x
                    for j in J:
                        for l in L:
                            final_alpha[j,l,w] = -alpha_bar[j,l].x/delta_bar.x 
                    
                    m_subp.remove(cs2_11)
                    m_subp.update()
                
                # Subproblem LP z was cont. -----------------------------------------------------------------------------------------------------    
                else:
                    # Subproblem LP z = 0 -----------------------------------------------------------------------------------------------------
                    cs2_11 = m_subp.addConstr(z <= 0)
                    m_subp.update()
                    m_subp.optimize()
                    if m_subp.status != GRB.OPTIMAL:
                        print('error! subproblem z = 0 not solved to optimality',m_subp.status)
                        exit()
    
                    obj_spz0 = m_subp.objVal      
                    get_duals()
                    #dualcs2_11 = cs2_11.pi                    
                    t_alpha0, t_beta0 = coeff_benders(dualcs2_1, dualcs2_2, dualcs2_3, dualcs2_4, dualcs2_5, dualcs2_6, dualcs2_7, dualcs2_8, 
                                             dualcs2_9, dualcs2_10, d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, w)
                    #t_beta0 = t_beta0 + dualcs2_11*(0)
                    m_subp.remove(cs2_11)
                    
                    # Subproblem LP z = 1 -----------------------------------------------------------------------------------------------------
                    cs2_11 = m_subp.addConstr(-z <= -1)
                    m_subp.update()
                    m_subp.optimize()
                
                    if m_subp.status == GRB.OPTIMAL:      
                        obj_spz1 = m_subp.objVal
                        get_duals()          
                        dualcs2_11 = cs2_11.pi
                        t_alpha1, t_beta1 = coeff_benders(dualcs2_1, dualcs2_2, dualcs2_3, dualcs2_4, dualcs2_5, dualcs2_6, dualcs2_7, dualcs2_8, 
                                                dualcs2_9, dualcs2_10, d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, w) 
                        t_beta1 = t_beta1 + dualcs2_11*(-1)
                        subp_obj[w] = max(obj_spz0,obj_spz1)
                        Qip += subp_obj[w]*(1/N)
                        
                        # generate a valid cut for the union -----------------------------------------------------------------------------------
                        if tight_cut is False:
                            theta_scenarioval[w] = thetaval
                        else:
                            theta_scenarioval[w] = beta_tight[w] + sum(alpha_tight[j,l,w]*yval[j,l] for j in J for l in L)   
                        
                        delta_bar.Obj = theta_scenarioval[w]
                        update_constraints_coe_cglp(t_beta0, t_beta1, t_alpha0, t_alpha1)
                        m_disj.optimize()
                        if delta_bar.x == 0:
                            print("error! separation problem gave a zero multiplier delta_bar for theta")
                            exit()
                        
                        final_beta[w] = beta_bar.x/delta_bar.x
                        for j in J:
                            for l in L:
                                final_alpha[j,l,w] = -alpha_bar[j,l].x/delta_bar.x           
                    else:
                        subp_obj[w] = obj_spz0
                        Qip += subp_obj[w]*(1/N)
                        final_beta[w] = t_beta0
                        for j in J:
                            for l in L:
                                final_alpha[j,l,w] = t_alpha0[j,l]
                
                    m_subp.remove(cs2_11)
                    m_subp.update()
                
            if (thetaval - Qip) / max(Qip, 1e-5) > 1e-5:
                model._cut_coef_list.append((final_beta, final_alpha))          # store final cut coeffs.
                model.cbLazy(model._theta <= (1/N)*sum(final_beta[w]+sum(sum(final_alpha[j,l,w]*model._y[j,l] for j in J) for l in L)   # add disjunctive cut
                                                           for w in range(N)))
                model.update()
                
#%% Run algorithm
m_mp = create_mp()
setattr(m_mp, '_cut_coef_list', [])     # list to store cut coefficients added to master problem
m_mp.setParam("OutputFlag", 1)          
m_subp.setParam("OutputFlag", 0)
m_disj.setParam("OutputFlag", 0)
m_mp.Params.LazyConstraints = 1
m_mp.setParam('TimeLimit',3600)

ini_time_OADC = time.time()
m_mp.optimize(callback)
timeOADC = time.time()-ini_time_OADC        # Optimization time
timeADC = time.time()-ini_time_ADC          # Optimization + setup of models

#%% Results
print('Opt+Setup Time ADC_V2', timeADC)
print('Optimization Time ADC_V2', timeOADC)    

OF = m_mp.objVal
ysol = {}
for j in J:
    for l in L:
        ysol[j,l] = m_mp._y[j,l].x
        if m_mp._y[j,l].x > 0.9:
            print("{}{}:".format(j,l),m_mp._y[j,l].x)
            
Fixed_costs = sum(ysol[j,l]*c_f[j,l] for j in J for l in L)
Second_Stage_OF = OF + Fixed_costs

nodes_explored = m_mp.NodeCount                                 
gap = m_mp.MIPGap                                               

# Save the statistics and results
with open("gurobi_statistics_ADCV2.txt", "w") as file:
    file.write("Scenarios,Method,Nodes Explored,Opt Time,Opt+Setup Time,Optimality Gap,Master Obj,Fixed Costs,2-Stage OF,r\n")
    file.write(f"{N},ADCV2,{nodes_explored},{timeOADC},{timeADC},{gap},{OF},{Fixed_costs},{Second_Stage_OF},{r}\n")

print("Objective Master:",OF)
print("Fixed Cost:", Fixed_costs) 
print("2-Stage OF:", Second_Stage_OF)
print("U", U)


