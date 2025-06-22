# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 21:19:23 2025

@author: Martha Sabogal
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time
from sys import exit
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

min_cp =  min(c_p[j] for j in J)
min_cs =  min(c_s[i] for i in I)    

U = {}
for w in range(N):
    U[w] = sum(d[k,w]*(p[k]-min_cp-min_cs) for k in K)   #upper bound for second stage obj. function
    
filename = f"gurobi_stats_ADCV2_M_C_II_{se}_{int(r*100)}.txt"
print(filename)

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
H = {0 , 1}                             # Union of polyhedra z=0 and z=1
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
    m_mp = gp.Model('Master_Multicut')
    
    # Decision variables
    m_mp._y = {}
    for j in J:
        for l in L:
            m_mp._y[j,l] = m_mp.addVar(vtype=GRB.BINARY, obj=-c_f[j,l])
            
    m_mp._theta = {}
    for w in range(N):        
        m_mp._theta[w] = m_mp.addVar(vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, ub = U[w], obj=1/N)
    
    # Objective Function 
    m_mp.ModelSense = GRB.MAXIMIZE
    
    # Constraints
    cm1_1 = {}
    for j in J:
        cm1_1[j] = m_mp.addConstr(gp.quicksum(m_mp._y[j,l] for l in L) <= 1)
        
    m_mp.update()
    return m_mp

# Update Subproblem RHS
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

# Create dual variables
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

#%% Algorithm

def callback(model, where):    
    if where == gp.GRB.Callback.MIPSOL:                                             # when find an integer solution
        
        yval = {}
        alpha_Qlp, beta_Qlp = {}, {}                                                # to store cut coeff.
        final_alpha, final_beta, subp_obj = {}, {}, {}
        
        for j in J:                                                                 # take variable solutions
            for l in L:
                yval[j, l] = model.cbGetSolution(model._y[j,l])
                if yval[j,l] > 0.9:
                    yval[j,l] = 1
                else:
                    yval[j,l] = 0 
                
        thetaval = {}
        for w in range(N):
            thetaval[w] = model.cbGetSolution(model._theta[w])
        
        update_obj_coe_cglp(yval)                                                   # update cglp obj function with y solution  

        for w in range(N):
            update_rhs(d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, yval, w)
            
            # Solve Subproblem as LP ---------------------------------------------------------------------------------------------------------------------
            m_subp.optimize()                   
            if m_subp.status != GRB.OPTIMAL:
                print("error! subproblem lp is not solved to optimality in scenario: ", w)
                exit()
            Qlp = m_subp.objVal
            get_duals()
            alpha_Qlp[w], beta_Qlp[w] = coeff_benders(dualcs2_1, dualcs2_2, dualcs2_3, dualcs2_4, dualcs2_5, dualcs2_6, dualcs2_7, dualcs2_8,
                                              dualcs2_9, dualcs2_10, d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, w)
            for j in J:
                for l in L:
                    if abs(alpha_Qlp[w][j,l]) < 1e-6: 
                        alpha_Qlp[w][j,l] = 0           
            beta_Qlp[w] = 0 if abs(beta_Qlp[w]) < 1e-6 else beta_Qlp[w]
            
            if (thetaval[w] - Qlp)/max(Qlp, 1e-5) > 1e-5:
                model.cbLazy(model._theta[w] <= beta_Qlp[w]+sum(sum(alpha_Qlp[w][j,l]*model._y[j,l] for j in J) for l in L))     # Benders cut
            
            else:
                if z.x <= 1e-5:                      # z lp was 0                
                    obj_spz0 = Qlp 
                    t_alpha0 = alpha_Qlp[w] 
                    t_beta0 = beta_Qlp[w]                
                    
                    # Solve Subproblem LP z = 1 ----------------------------------------------------------------------------------------------------------
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
                        
                        # generate a valid cut for the union -----------------------------------------------------------------------------------  
                        delta_bar.Obj = thetaval[w]
                        update_constraints_coe_cglp(t_beta0, t_beta1, t_alpha0, t_alpha1)
                        m_disj.optimize()
                        
                        if delta_bar.x == 0:
                            print("separation problem gave a zero multiplier delta_bar for theta in scenario", w)
                            m_subp.remove(cs2_11)
                            m_subp.update()
                            continue             # Skip the rest of the loop for this scenario and proceed with the next `w`
                    
                        final_beta[w] = 0 if abs(beta_bar.x/delta_bar.x) < 1e-6 else beta_bar.x/delta_bar.x
                        for j in J:
                            for l in L:
                                final_alpha[j,l,w] = 0 if abs(-alpha_bar[j,l].x/delta_bar.x) < 1e-6 else -alpha_bar[j,l].x/delta_bar.x         
                    else:
                        subp_obj[w] = obj_spz0
                        final_beta[w] = t_beta0
                        for j in J:
                            for l in L:
                                final_alpha[j,l,w] = t_alpha0[j,l]
                                
                    m_subp.remove(cs2_11)
                    m_subp.update()
                 
                elif z.x >= 1 - 1e-5:                # z lp was 1
                    obj_spz1 = Qlp 
                    t_alpha1 = alpha_Qlp[w] 
                    t_beta1 = beta_Qlp[w]    
                                
                    # Solve Subproblem LP z = 0 ------------------------------------------------------------------------------------------------------
                    cs2_11 = m_subp.addConstr(z <= 0)
                    m_subp.update()
                    m_subp.optimize()
                    if m_subp.status != GRB.OPTIMAL:
                        print('error! subproblem z = 0 not solved to optimality in scenario', w)
                        exit()
                    obj_spz0 = m_subp.objVal      
                    get_duals()
                    #dualcs2_11 = cs2_11.pi 
                    t_alpha0, t_beta0 = coeff_benders(dualcs2_1, dualcs2_2, dualcs2_3, dualcs2_4, dualcs2_5, dualcs2_6, dualcs2_7, dualcs2_8, 
                                                 dualcs2_9, dualcs2_10, d, q_s, q_p, gamma_s, gamma_p, eta, xi_nd, xi_qs, xi_qp, M3, w)
                    #t_beta0 = t_beta0 + dualcs2_11*(0)
                    subp_obj[w] = max(obj_spz0,obj_spz1)
                    
                    # generate a valid cut for the union -----------------------------------------------------------------------------------   
                    delta_bar.Obj = thetaval[w]
                    update_constraints_coe_cglp(t_beta0, t_beta1, t_alpha0, t_alpha1)
                    m_disj.optimize()
                    
                    if delta_bar.x == 0:
                        print("separation problem gave a zero multiplier delta_bar for theta in scenario", w)
                        m_subp.remove(cs2_11)
                        m_subp.update()
                        continue
                    
                    final_beta[w] = 0 if abs(beta_bar.x/delta_bar.x) < 1e-6 else beta_bar.x/delta_bar.x
                    for j in J:
                        for l in L:
                            final_alpha[j,l,w] = 0 if abs(-alpha_bar[j,l].x/delta_bar.x) < 1e-6 else -alpha_bar[j,l].x/delta_bar.x 
                    
                    m_subp.remove(cs2_11)
                    m_subp.update()
                    
                else:                                  # z lp was fractional  
                    # Subproblem LP z = 0 ---------------------------------------------------------------------------------------------------------
                    cs2_11 = m_subp.addConstr(z <= 0)
                    m_subp.update()
                    m_subp.optimize()
                    if m_subp.status != GRB.OPTIMAL:
                       print('error! subproblem z = 0 not solved to optimality in scenario', w)
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
                        
                        # generate a valid cut for the union -----------------------------------------------------------------------------------             
                        delta_bar.Obj = thetaval[w]
                        update_constraints_coe_cglp(t_beta0, t_beta1, t_alpha0, t_alpha1)
                        m_disj.optimize()
                    
                        if delta_bar.x == 0:
                            print("separation problem gave a zero multiplier delta_bar for theta in scenario", w)
                            m_subp.remove(cs2_11)
                            m_subp.update()
                            continue
                    
                        final_beta[w] = 0 if abs(beta_bar.x/delta_bar.x) < 1e-6 else beta_bar.x/delta_bar.x
                        for j in J:
                            for l in L:
                                final_alpha[j,l,w] = 0 if abs(-alpha_bar[j,l].x/delta_bar.x) < 1e-6 else -alpha_bar[j,l].x/delta_bar.x
                                
                    else:
                        subp_obj[w] = obj_spz0
                        final_beta[w] = 0 if abs(t_beta0) < 1e-6 else t_beta0
                        for j in J:
                            for l in L:
                                final_alpha[j,l,w] = 0 if abs(t_alpha0[j,l]) < 1e-6 else t_alpha0[j,l]
                           
                    m_subp.remove(cs2_11)
                    m_subp.update()
                
                if (thetaval[w] - subp_obj[w])/max(subp_obj[w], 1e-5) > 1e-5:
                        model.cbLazy(model._theta[w] <= final_beta[w]+sum(sum(final_alpha[j,l,w]*model._y[j,l] for j in J) for l in L))  # Disjunctive cut

#%% Run algorithm
m_mp = create_mp()
m_mp.setParam("OutputFlag", 1)          
m_subp.setParam("OutputFlag", 0)
m_disj.setParam("OutputFlag", 0)
m_mp.Params.LazyConstraints = 1
m_mp.setParam('TimeLimit',3600)

m_mp.setParam("Threads", 1)
m_subp.setParam("Threads", 1)
m_disj.setParam("Threads", 1)
 
ini_time_OADC = time.time()
m_mp.optimize(callback)
timeOADC = time.time()-ini_time_OADC        # Optimization time
timeADC = time.time()-ini_time_ADC          # Optimization + setup of models

#%% Results
print('Opt+Setup Time ADC_V2_M', timeADC)
print('Optimization Time ADC_V2_M', timeOADC)    

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
with open(filename, "w") as file:
    file.write("Scenarios,Method,Nodes,Opt Time,Opt+Setup Time,Gap,Master Obj,Fixed Costs,2-Stage OF,r,seed,status\n")
    file.write(f"{N},ADCV2_M_C_II,{nodes_explored},{timeOADC},{timeADC},{gap},{OF},{Fixed_costs},{Second_Stage_OF},{r},{se},{m_mp.status}\n")

print("Objective Master:",OF)
print("Fixed Cost:", Fixed_costs) 
print("2-Stage OF:", Second_Stage_OF)