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
import copy
from sklearn.linear_model import Ridge

def funct1(X):
    if len(X.shape) == 2:
        X1 = X[:,0]; X2 = X[:,1]
    elif len(X.shape) == 1:
        X1 = X[0]; X2 = X[1]
    y = X1 + 2*X2 
    return y 

def funct2(X):
    if len(X.shape) == 2:
        X1 = X[:,0]; X2 = X[:,1]
    elif len(X.shape) == 1:
        X1 = X[0]; X2 = X[1]
    y = 1.2*X1 + 2.4*X2 
    return y 

funct_ls = [funct1,funct2]

def cal_regret(est_mc,eval_optimal,X_test,n_eval):
    arm = np.argmax(est_mc,0)
    eval_mc = []
    for i in range(n_eval):
        eval_mc.append(funct_ls[arm[i]](X_test[i]))
    regret_mc = np.sum(eval_optimal) -  np.sum(eval_mc)
    return regret_mc

def sim_mab(seed,n_arms,w_dim,N,percent,n_eval,noise):
    np.random.seed(seed)
    X_ls = []
    for i in range(n_arms):
        X_ls.append([])
    for i in range(N):
        x = np.random.normal(0,1,w_dim)
        if np.random.uniform(0,1,1)[0] < percent:
            index = np.argmax([funct(x) for funct in funct_ls])
            X_ls[index].append(x) 
        else:
            index = np.argmin([funct(x) for funct in funct_ls])
            X_ls[index].append(x)    
    X_ls = [np.array(X) for X in X_ls]
    X_test = np.random.normal(0,1,n_eval*w_dim).reshape([-1,w_dim])
    rbf_feature = RBFSampler(gamma=1, random_state=1,n_components=n_components)
    X_test_basis = rbf_feature.fit_transform(X_test)

    est_mc_non_pessi_ls = []
    est_mc_pessi_ls = []
    est_mc_quantile_ls = []
    est_mc_uniform_ls = []
    for alpha in [0.9,0.95,0.99]:
        critical_value = scipy.stats.chi2.ppf(np.sqrt(alpha), X_test_basis.shape[1])

        est_mc_non_pessi = []
        est_mc_pessi = []
        est_mc_quantile = []
        est_mc_uniform = []
        est_mc_bound_mean = []
        est_mc_bound_std = []
        bound_param_ls = []
        for i in range(n_arms):
            X_train = X_ls[i]
            funct = funct_ls[i]
            y_train = funct(X_train) + np.random.randn(X_train.shape[0]) * noise

            X_basis = rbf_feature.fit_transform(X_train)

            bound_param = w_dim * np.sqrt(np.log(2*w_dim*X_train.shape[0]/np.sqrt(1-(1-alpha)/2)))
            bound_param_ls.append(bound_param)

            reg = Ridge(fit_intercept=False,alpha=1).fit(X_basis, y_train)
            bound_mean = reg.predict(X_test_basis)
            Lambda = np.matmul(X_basis.T,X_basis) + np.diag(np.ones(X_basis.shape[1]))
            try:
                bound_std = np.sqrt(np.diag(np.matmul(np.matmul(X_test_basis,inv(Lambda)),X_test_basis.T)))
            except:
                bound_std = np.sqrt(np.diag(np.matmul(np.matmul(X_test_basis,pinv(Lambda)),X_test_basis.T)))

            br = BayesianRidge(fit_intercept=False, tol=1e-5, verbose=False)
            br.fit(X_basis,y_train)
            MC_mean, MC_std = br.predict(X_test_basis, return_std=True)
            MC_quantile = scipy.stats.norm.ppf(1-np.sqrt(1-(1-alpha)/2), loc=MC_mean, scale=MC_std)

            n_sampling = 10000
            m_i  = br.coef_
            cov_i = br.sigma_
            w_sample = np.random.multivariate_normal(m_i,cov_i,n_sampling).T
            est_sample = np.matmul(X_test_basis,w_sample)
            v = w_sample - m_i.reshape([-1,1])
            try:
                test_stat = np.diag(np.matmul(np.matmul(v.T,inv(cov_i)),v))
            except:
                test_stat = np.diag(np.matmul(np.matmul(v.T,pinv(cov_i)),v))
            accept = test_stat < critical_value
            est_uniform_i = np.min(est_sample[:,accept],1)

            est_mc_non_pessi_i = MC_mean
            est_mc_pessi_i = MC_mean - scipy.stats.norm.ppf(np.sqrt(1-(1-alpha)/2))*MC_std
            est_mc_quantile_i = MC_quantile  

            est_mc_non_pessi.append(est_mc_non_pessi_i)
            est_mc_pessi.append(est_mc_pessi_i)
            est_mc_quantile.append(est_mc_quantile_i)
            est_mc_uniform.append(est_uniform_i)
            est_mc_bound_mean.append(bound_mean)
            est_mc_bound_std.append(bound_std)
                                               
        est_mc_non_pessi_ls.append(est_mc_non_pessi)
        est_mc_pessi_ls.append(est_mc_pessi)
        est_mc_quantile_ls.append(est_mc_quantile)
        est_mc_uniform_ls.append(est_mc_uniform)                                           

    eval_optimal = []
    for i in range(n_eval):
        eval_optimal.append(np.max([funct(X_test[i]) for funct in funct_ls]))

    regret_mc_non_pessi = [cal_regret(item,eval_optimal,X_test,n_eval) for item in est_mc_non_pessi_ls]
    regret_mc_pessi = [cal_regret(item,eval_optimal,X_test,n_eval) for item in est_mc_pessi_ls]
    regret_mc_quantile = [cal_regret(item,eval_optimal,X_test,n_eval) for item in est_mc_quantile_ls]
    regret_mc_uniform = [cal_regret(item,eval_optimal,X_test,n_eval) for item in est_mc_uniform_ls]

    regret_mc_bound = []
    for c in c_ls:
        est_mc_bound = []
        for k in range(n_arms):
            est_mc_bound.append(est_mc_bound_mean[k] - c*bound_param_ls[k]*est_mc_bound_std[k])
        regret_mc_bound.append(cal_regret(est_mc_bound,eval_optimal,X_test,n_eval))
    return regret_mc_non_pessi, regret_mc_pessi, regret_mc_quantile, regret_mc_uniform,regret_mc_bound 

def generate_result(n_arms,w_dim,N,percent,n_eval,noise):
    regret_non_pessi_ls = []
    regret_pessi_ls = []
    regret_quantile_ls = [] 
    regret_uniform_ls = [] 
    regret_bound_ls = [] 

    for seed in tqdm(range(50)):
        regret_non_pessi, regret_pessi, regret_quantile, regret_uniform, regret_bound = sim_mab(seed,n_arms,w_dim,N,percent,n_eval,noise)

        regret_non_pessi_ls.append(regret_non_pessi)
        regret_pessi_ls.append(regret_pessi)
        regret_quantile_ls.append(regret_quantile)
        regret_uniform_ls.append(regret_uniform)
        regret_bound_ls.append(regret_bound)

    out = [regret_non_pessi_ls, regret_pessi_ls, regret_quantile_ls, regret_uniform_ls, regret_bound_ls]
    return out

c_ls = [0,0.01,0.1,0.5,1,2,5,10]   

string = "toy"
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.95
    noise = 0.1
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.5
    noise = 0.1
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.95
    noise = 0.5
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.5
    noise = 0.5
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.95
    noise = 1
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.5
    noise = 1
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.95
    noise = 2
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.5
    noise = 2
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.95
    noise = 5
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.5
    noise = 5
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.95
    noise = 10
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.5
    noise = 10
    n_eval = 500
    w_dim = 2
    n_components = 100
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent)+"_noise_"+str(noise),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent)+"_noise_"+str(noise),N_arr)