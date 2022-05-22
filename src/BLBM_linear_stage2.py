import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
import numpy as np
import random
import scipy
import scipy.stats
from numpy.linalg import inv
from numpy.linalg import pinv
import copy
from sklearn.linear_model import Ridge

def funct1(X):
    if len(X.shape) == 2:
        X1 = X[:,0]; X2 = X[:,1]; X3 = X[:,2]
    elif len(X.shape) == 1:
        X1 = X[0]; X2 = X[1]; X3 = X[2] 
    y = 0.2*X1 + 0.25*X2 + 0.3*X3
    return y 

def funct2(X):
    if len(X.shape) == 2:
        X1 = X[:,0]; X2 = X[:,1]; X3 = X[:,2]
    elif len(X.shape) == 1:
        X1 = X[0]; X2 = X[1]; X3 = X[2] 
    y = 0.25*X1 + 0.3*X2 + 0.35*X3
    return y 

funct_ls = [funct1,funct2]

def sim_mab(seed,N,s_dim1,s_dim2,percent,noise,n_eval,W1_ls):
    np.random.seed(seed)
    action_space = [[0,0],[0,1],[1,0],[1,1]]
    a1_ls = []
    S1_ls = []
    S2_ls = []
    reward_ls = []
    for i in range(4):
        a1_ls.append([])
        S1_ls.append([])
        S2_ls.append([])
        reward_ls.append([])
    for i in range(N):
        s1 = np.random.normal(0,1,s_dim1).reshape([1,-1])
        r_max_ls = []
        for j in range(2):
            W1 = W1_ls[j]
            S20 = np.matmul(s1,W1) + np.random.normal(0,1,s_dim2*500).reshape([-1,s_dim2])
            r_ls = []
            for k in range(2):
                funct = funct_ls[k]
                r = np.mean(funct(S20))
                r_ls.append(r)  
            r_max = np.max(r_ls)
            r_max_ls.append(r_max)
        if np.random.uniform(0,1,1)[0] < percent:
            a1 = np.argmax(r_max_ls)
        else:
            a1 = np.argmin(r_max_ls)

        W1 = W1_ls[a1]
        s2 = np.matmul(s1,W1).flatten() +  np.random.normal(0,1,s_dim2)

        r_ls = []
        for j in range(2):
            funct = funct_ls[j]
            r = funct(s2)
            r_ls.append(r)       

        if np.random.uniform(0,1,1)[0] < percent:
            a2 = np.argmax(r_ls)
        else:
            a2 = np.argmin(r_ls)
        index = a1*2 + a2
        S1_ls[index].append(s1[0])    
        a1_ls[index].append(a1) 
        S2_ls[index].append(s2)   
        reward_ls[index].append(r_ls[a2]+np.random.randn(1)[0] * noise)    

    a1_ls = [np.array(A).reshape([-1,1]) for A in a1_ls]  
    S1_ls = [np.array(S) for S in S1_ls]     
    S2_ls = [np.array(S) for S in S2_ls]  
    reward_ls = [np.array(reward) for reward in reward_ls]

    S1_test = np.random.normal(0,1,n_eval*s_dim1).reshape([-1,s_dim1])
    rbf_feature = RBFSampler(gamma=1, random_state=1,n_components=500)
    S1_test_basis = rbf_feature.fit_transform(S1_test)
    critical_value1 = scipy.stats.chi2.ppf(1-(1-(1-(1-0.95)/2))/2, S1_test_basis.shape[1])
    critical_value2 = scipy.stats.chi2.ppf(np.sqrt(1-(1-0.95)/2), S1_test_basis.shape[1])

    select_stage1 = [[0,1],[2,3]]
    select_stage2 = [[0,2],[1,3]]
    br2_ls = []
    reg2_ls = []
    Lambda2_ls = []
    bound_param2_ls = []
    for select in select_stage2:
        i1, i2 = select
        A1 = np.concatenate([a1_ls[i1],a1_ls[i2]]) 
        S1 = np.concatenate([S1_ls[i1],S1_ls[i2]]) 
        S2 = np.concatenate([S2_ls[i1],S2_ls[i2]]) 
        S2_input = np.concatenate([A1,S1,S2],axis=1)
        S2_basis = rbf_feature.fit_transform(S2_input)
        r = np.concatenate([reward_ls[i1],reward_ls[i2]]) 
        br2 = BayesianRidge(fit_intercept=False, tol=1e-5, verbose=False)
        br2.fit(S2_basis,r)
        br2_ls.append(copy.deepcopy(br2))

        reg2 = Ridge(fit_intercept=False,alpha=1).fit(S2_basis,r)
        Lambda2 = np.matmul(S2_basis.T,S2_basis) + np.diag(np.ones(S2_basis.shape[1]))
        reg2_ls.append(copy.deepcopy(reg2))
        Lambda2_ls.append(Lambda2)
        bound_param2 = s_dim2 * np.sqrt(np.log(2*s_dim2*S2_basis.shape[0]/np.sqrt(1-(1-0.975)/2)))
        bound_param2_ls.append(bound_param2)

    br1_non_pessi_ls = []
    br1_pessi_ls = []
    br1_quantile_ls = []
    br1_uniform_ls = []
    reg1_ls = []
    Lambda1_ls = []
    bound_param1_ls = []
    for select in select_stage1:
        i1, i2 = select
        S1 = np.concatenate([S1_ls[i1],S1_ls[i2]]) 
        S1_basis = rbf_feature.fit_transform(S1)

        A1 = np.concatenate([a1_ls[i1],a1_ls[i2]]) 
        S1 = np.concatenate([S1_ls[i1],S1_ls[i2]]) 
        S2 = np.concatenate([S2_ls[i1],S2_ls[i2]]) 
        S2_input = np.concatenate([A1,S1,S2],axis=1)
        S2_basis = rbf_feature.fit_transform(S2_input)    

        est_mc_non_pessi = []
        est_mc_pessi = []
        est_mc_quantile = []
        est_mc_uniform = []
        est_mc_bound_mean = []
        est_mc_bound_std = []
        for i in range(2):
            MC_mean, MC_std = br2_ls[i].predict(S2_basis, return_std=True)
            MC_quantile = scipy.stats.norm.ppf(1-np.sqrt(1-(1-0.975)/2), loc=MC_mean, scale=MC_std)       

            n_sampling = 10000
            m_i  = br2_ls[i].coef_
            cov_i = br2_ls[i].sigma_
            w_sample = np.random.multivariate_normal(m_i,cov_i,n_sampling).T
            est_sample = np.matmul(S2_basis,w_sample)
            v = w_sample - m_i.reshape([-1,1])
            try:
                test_stat = np.diag(np.matmul(np.matmul(v.T,inv(cov_i)),v))
            except:
                test_stat = np.diag(np.matmul(np.matmul(v.T,pinv(cov_i)),v))
            accept = test_stat < critical_value2
            est_uniform_i = np.min(est_sample[:,accept],1)
  
            est_mc_non_pessi_i = MC_mean
            est_mc_pessi_i = MC_mean - scipy.stats.norm.ppf(np.sqrt(1-(1-0.975)/2))*MC_std
            est_mc_quantile_i = MC_quantile  

            est_mc_non_pessi.append(est_mc_non_pessi_i)
            est_mc_pessi.append(est_mc_pessi_i)
            est_mc_quantile.append(est_mc_quantile_i)
            est_mc_uniform.append(est_uniform_i)

            bound_mean = reg2_ls[i].predict(S2_basis)
            try:
                bound_std = np.sqrt(np.diag(np.matmul(np.matmul(S2_basis,inv(Lambda2_ls[i])),S2_basis.T)))
            except:
                bound_std = np.sqrt(np.diag(np.matmul(np.matmul(S2_basis,pinv(Lambda2_ls[i])),S2_basis.T)))
            est_mc_bound_mean.append(bound_mean)
            est_mc_bound_std.append(bound_std)

        reward_non_pessi = np.max(est_mc_non_pessi,0)
        reward_pessi = np.max(est_mc_pessi,0)
        reward_quantile = np.max(est_mc_quantile,0)
        reward_uniform = np.max(est_mc_uniform,0)

        reward_bound_ls = []
        for c in c_ls:
            est_mc_bound = []
            for m in range(2):
                est_mc_bound.append(est_mc_bound_mean[m] - c*bound_param2_ls[m]*est_mc_bound_std[m])
            reward_bound = np.max(est_mc_bound,0)
            reward_bound_ls.append(reward_bound)

        br1_non_pessi = BayesianRidge(fit_intercept=False, tol=1e-5, verbose=False)
        br1_non_pessi.fit(S1_basis,reward_non_pessi)   
        br1_non_pessi_ls.append(copy.deepcopy(br1_non_pessi))

        br1_pessi = BayesianRidge(fit_intercept=False, tol=1e-5, verbose=False)
        br1_pessi.fit(S1_basis,reward_pessi)   
        br1_pessi_ls.append(copy.deepcopy(br1_pessi))

        br1_quantile = BayesianRidge(fit_intercept=False, tol=1e-5, verbose=False)
        br1_quantile.fit(S1_basis,reward_quantile)   
        br1_quantile_ls.append(copy.deepcopy(br1_quantile))

        br1_uniform = BayesianRidge(fit_intercept=False, tol=1e-5, verbose=False)
        br1_uniform.fit(S1_basis,reward_uniform)   
        br1_uniform_ls.append(copy.deepcopy(br1_uniform))

        reg1_c = []
        Lambda1_c = []
        for m in range(len(c_ls)):
            reg1 = Ridge(fit_intercept=False,alpha=1).fit(S1_basis,reward_bound_ls[m])
            Lambda1 = np.matmul(S1_basis.T,S1_basis) + np.diag(np.ones(S1_basis.shape[1]))
            reg1_c.append(reg1)
            Lambda1_c.append(Lambda1)
        reg1_ls.append(copy.deepcopy(reg1_c))
        Lambda1_ls.append(Lambda1_c)
        bound_param1 = s_dim1 * np.sqrt(np.log(2*s_dim1*S1_basis.shape[0]/(1-(1-(1-(1-0.975)/2))/2)))
        bound_param1_ls.append(bound_param1)

    est_mc_non_pessi = []
    est_mc_pessi = []
    est_mc_quantile = []
    est_mc_uniform = []
    est_mc_bound_ls = []
    for i in range(2):
        MC_mean, MC_std = br1_non_pessi_ls[i].predict(S1_test_basis, return_std=True)
        est_mc_non_pessi.append(MC_mean)
        MC_mean, MC_std = br1_pessi_ls[i].predict(S1_test_basis, return_std=True)
        est_mc_pessi.append(MC_mean - scipy.stats.norm.ppf(1-(1-(1-(1-0.975)/2))/2)*MC_std)
        MC_mean, MC_std = br1_quantile_ls[i].predict(S1_test_basis, return_std=True)
        MC_quantile = scipy.stats.norm.ppf((1-(1-(1-0.975)/2))/2, loc=MC_mean, scale=MC_std)
        est_mc_quantile.append(MC_quantile)

        n_sampling = 10000
        m_i  = br1_uniform_ls[i].coef_
        cov_i = br1_uniform_ls[i].sigma_
        w_sample = np.random.multivariate_normal(m_i,cov_i,n_sampling).T
        est_sample = np.matmul(S1_test_basis,w_sample)
        v = w_sample - m_i.reshape([-1,1])
        try:
            test_stat = np.diag(np.matmul(np.matmul(v.T,inv(cov_i)),v))
        except:
            test_stat = np.diag(np.matmul(np.matmul(v.T,pinv(cov_i)),v))
        accept = test_stat < critical_value1
        est_uniform_i = np.min(est_sample[:,accept],1)

        est_mc_uniform.append(est_uniform_i)

        est_mc_bound = []
        for m in range(len(c_ls)):
            bound_mean = reg1_ls[i][m].predict(S1_test_basis)
            try:
                bound_std = np.sqrt(np.diag(np.matmul(np.matmul(S1_test_basis,inv(Lambda1_ls[i][m])),S1_test_basis.T)))
            except:
                bound_std = np.sqrt(np.diag(np.matmul(np.matmul(S1_test_basis,pinv(Lambda1_ls[i][m])),S1_test_basis.T)))
            est_mc_bound.append(bound_mean - c_ls[m]*bound_param1_ls[i]*bound_std)
        est_mc_bound_ls.append(est_mc_bound)

    arm1_non_pessi = np.argmax(est_mc_non_pessi,0)
    arm1_pessi = np.argmax(est_mc_pessi,0)
    arm1_quantile = np.argmax(est_mc_quantile,0)
    arm1_uniform = np.argmax(est_mc_uniform,0)

    arm1_bound_ls = []
    for q in range(len(c_ls)):
        est_mc_bound = []
        for m in range(2):
            est_mc_bound.append(est_mc_bound_ls[m][q])
        arm1_bound = np.argmax(est_mc_bound,0)
        arm1_bound_ls.append(arm1_bound)

    S2_test_non_pessi = np.array([np.matmul(S1_test[i].reshape([1,-1]),W1_ls[arm1_non_pessi[i]]).flatten() for i in range(n_eval)])
    S2_test_pessi = np.array([np.matmul(S1_test[i].reshape([1,-1]),W1_ls[arm1_pessi[i]]).flatten() for i in range(n_eval)])
    S2_test_quantile = np.array([np.matmul(S1_test[i].reshape([1,-1]),W1_ls[arm1_quantile[i]]).flatten() for i in range(n_eval)])
    S2_test_uniform = np.array([np.matmul(S1_test[i].reshape([1,-1]),W1_ls[arm1_uniform[i]]).flatten() for i in range(n_eval)])

    S2_test_bound_ls = []
    for m in range(len(c_ls)):
        S2_test_bound = np.array([np.matmul(S1_test[i].reshape([1,-1]),W1_ls[arm1_bound_ls[m][i]]).flatten() for i in range(n_eval)])
        S2_test_bound_ls.append(S2_test_bound)

    est_mc_non_pessi = []
    est_mc_pessi = []
    est_mc_quantile = []
    est_mc_uniform = []
    est_mc_bound_ls = []
    for i in range(2):
        S2_input = np.concatenate([arm1_non_pessi.reshape([-1,1]),S1_test,S2_test_non_pessi],axis=1)
        MC_mean, MC_std = br2_ls[i].predict(rbf_feature.fit_transform(S2_input), return_std=True)
        est_mc_non_pessi.append(MC_mean)

        S2_input = np.concatenate([arm1_pessi.reshape([-1,1]),S1_test,S2_test_pessi],axis=1)
        MC_mean, MC_std = br2_ls[i].predict(rbf_feature.fit_transform(S2_input), return_std=True)
        est_mc_pessi.append(MC_mean - scipy.stats.norm.ppf(np.sqrt(1-(1-0.975)/2))*MC_std)

        S2_input = np.concatenate([arm1_quantile.reshape([-1,1]),S1_test,S2_test_quantile],axis=1)
        MC_mean, MC_std = br2_ls[i].predict(rbf_feature.fit_transform(S2_input), return_std=True)
        MC_quantile = scipy.stats.norm.ppf(1-np.sqrt(1-(1-0.975)/2), loc=MC_mean, scale=MC_std)
        est_mc_quantile.append(MC_quantile)

        n_sampling = 10000
        m_i  = br2_ls[i].coef_
        cov_i = br2_ls[i].sigma_
        S2_input = np.concatenate([arm1_uniform.reshape([-1,1]),S1_test,S2_test_uniform],axis=1)
        S2_test_uniform_basis = rbf_feature.fit_transform(S2_input)
        w_sample = np.random.multivariate_normal(m_i,cov_i,n_sampling).T
        est_sample = np.matmul(S2_test_uniform_basis,w_sample)
        v = w_sample - m_i.reshape([-1,1])
        try:
            test_stat = np.diag(np.matmul(np.matmul(v.T,inv(cov_i)),v))
        except:
            test_stat = np.diag(np.matmul(np.matmul(v.T,pinv(cov_i)),v))
        accept = test_stat < critical_value2
        est_uniform_i = np.min(est_sample[:,accept],1)
        est_mc_uniform.append(est_uniform_i)

        est_mc_bound = []
        for m in range(len(c_ls)):
            S2_input = np.concatenate([arm1_bound_ls[m].reshape([-1,1]),S1_test,S2_test_bound_ls[m]],axis=1)
            S2_test_basis = rbf_feature.fit_transform(S2_input)
            bound_mean = reg2_ls[i].predict(S2_test_basis)
            try:
                bound_std = np.sqrt(np.diag(np.matmul(np.matmul(S2_test_basis,inv(Lambda2_ls[i])),S2_test_basis.T)))
            except:
                bound_std = np.sqrt(np.diag(np.matmul(np.matmul(S2_test_basis,pinv(Lambda2_ls[i])),S2_test_basis.T)))
            est_mc_bound.append(bound_mean - c_ls[m]*bound_param1_ls[i]*bound_std)
        est_mc_bound_ls.append(est_mc_bound)

    arm2_non_pessi = np.argmax(est_mc_non_pessi,0)
    arm2_pessi = np.argmax(est_mc_pessi,0)
    arm2_quantile = np.argmax(est_mc_quantile,0)
    arm2_uniform = np.argmax(est_mc_uniform,0)

    arm2_bound_ls = []
    for q in range(len(c_ls)):
        est_mc_bound = []
        for m in range(2):
            est_mc_bound.append(est_mc_bound_ls[m][q])
        arm2_bound = np.argmax(est_mc_bound,0)
        arm2_bound_ls.append(arm2_bound)

    reward_non_pessi = np.array([funct_ls[arm2_non_pessi[i]](S2_test_non_pessi[i]).flatten() for i in range(n_eval)]).flatten()
    reward_pessi = np.array([funct_ls[arm2_pessi[i]](S2_test_pessi[i]).flatten() for i in range(n_eval)]).flatten()
    reward_quantile = np.array([funct_ls[arm2_quantile[i]](S2_test_quantile[i]).flatten() for i in range(n_eval)]).flatten()
    reward_uniform = np.array([funct_ls[arm2_uniform[i]](S2_test_uniform[i]).flatten() for i in range(n_eval)]).flatten()

    reward_bound_ls = []
    for m in range(len(c_ls)):
        reward_bound = np.array([funct_ls[arm2_bound_ls[m][i]](S2_test_bound_ls[m][i]).flatten() for i in range(n_eval)]).flatten()
        reward_bound_ls.append(reward_bound)

    r_ls = []
    for i in range(4):
        a1, a2 = action_space[i]
        W1 = W1_ls[a1]
        funct = funct_ls[a2]
        r = funct(np.matmul(S1_test,W1)).flatten()
        r_ls.append(r)
    r_ls = np.array(r_ls).T
    reward_optimal = np.max(r_ls,1)

    regret_mc_non_pessi = np.sum(reward_optimal) - np.sum(reward_non_pessi)
    regret_mc_pessi = np.sum(reward_optimal) - np.sum(reward_pessi)
    regret_mc_quantile = np.sum(reward_optimal) - np.sum(reward_quantile)
    regret_mc_uniform = np.sum(reward_optimal) - np.sum(reward_uniform)
    
    regret_mc_bound = []
    for m in range(len(c_ls)):
        regret_bound = np.sum(reward_optimal) - np.sum(reward_bound_ls[m])
        regret_mc_bound.append(regret_bound)

    return regret_mc_non_pessi, regret_mc_pessi, regret_mc_quantile, regret_mc_uniform,regret_mc_bound 

def generate_result(N,s_dim1,s_dim2,percent,noise,n_eval,W1_ls):
    regret_non_pessi_ls = []
    regret_pessi_ls = []
    regret_quantile_ls = [] 
    regret_uniform_ls = [] 
    regret_bound_ls = [] 

    for seed in tqdm(range(50)):
        try:
            regret_non_pessi, regret_pessi, regret_quantile, regret_uniform, regret_bound = sim_mab(seed,N,s_dim1,s_dim2,percent,noise,n_eval,W1_ls)

            regret_non_pessi_ls.append(regret_non_pessi)
            regret_pessi_ls.append(regret_pessi)
            regret_quantile_ls.append(regret_quantile)
            regret_uniform_ls.append(regret_uniform)
            regret_bound_ls.append(regret_bound)
        except:
            print("revise")

    out = [regret_non_pessi_ls, regret_pessi_ls, regret_quantile_ls, regret_uniform_ls, regret_bound_ls]
    return out

    
c_ls = [0,0.01,0.1,0.5,1,2,5,10]   
string = "blbm_linear_stage2"

info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    np.random.seed(0)
    s_dim1 = 2
    s_dim2 = 3
    percent = 0.95
    noise = 0.1
    n_eval = 500
    W1_arm1 = np.random.normal(0,1,s_dim1*s_dim2).reshape([s_dim1,s_dim2])
    W1_arm2 = W1_arm1 + 0.05
    W1_ls = [W1_arm1, W1_arm2]
    info = generate_result(N,s_dim1,s_dim2,percent,noise,n_eval,W1_ls)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    np.random.seed(0)
    s_dim1 = 2
    s_dim2 = 3
    percent = 0.85
    noise = 0.1
    n_eval = 500
    W1_arm1 = np.random.normal(0,1,s_dim1*s_dim2).reshape([s_dim1,s_dim2])
    W1_arm2 = W1_arm1 + 0.05
    W1_ls = [W1_arm1, W1_arm2]
    info = generate_result(N,s_dim1,s_dim2,percent,noise,n_eval,W1_ls)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    np.random.seed(0)
    s_dim1 = 2
    s_dim2 = 3
    percent = 0.75
    noise = 0.1
    n_eval = 500
    W1_arm1 = np.random.normal(0,1,s_dim1*s_dim2).reshape([s_dim1,s_dim2])
    W1_arm2 = W1_arm1 + 0.05
    W1_ls = [W1_arm1, W1_arm2]
    info = generate_result(N,s_dim1,s_dim2,percent,noise,n_eval,W1_ls)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent),N_arr)

info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    np.random.seed(0)
    s_dim1 = 2
    s_dim2 = 3
    percent = 0.5
    noise = 0.1
    n_eval = 500
    W1_arm1 = np.random.normal(0,1,s_dim1*s_dim2).reshape([s_dim1,s_dim2])
    W1_arm2 = W1_arm1 + 0.05
    W1_ls = [W1_arm1, W1_arm2]
    info = generate_result(N,s_dim1,s_dim2,percent,noise,n_eval,W1_ls)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent),N_arr)