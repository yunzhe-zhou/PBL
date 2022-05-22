import math
import torch.optim as optim
from BayesBackpropagation import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import copy
from numpy.linalg import inv
from sklearn.utils import shuffle

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

# Define training step for regression
def train(net, optimizer, data, target, NUM_BATCHES):
    for i in range(NUM_BATCHES):
        net.zero_grad()
        x = data[i]
        y = target[i].reshape((-1,1))
        loss = net.BBB_loss(x, y)
        loss.backward()
        optimizer.step()

#Data Generation step
if torch.cuda.is_available():
    Var = lambda x, dtype=torch.cuda.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor
else:
    Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor

def Bayesian_NN_est(X,y,N):
    #Hyperparameter setting
    TRAIN_EPOCHS = 500
    SAMPLES = 5
    TEST_SAMPLES = 10000
    BATCH_SIZE = 100
    NUM_BATCHES = np.int64(X.shape[0]/BATCH_SIZE+1)
    CLASSES = 1
    PI = 0.25
    SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
    SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)

    d = X.shape[1]

    X_rep = np.concatenate([X,X],axis=0)
    y_rep = np.concatenate([y,y],axis=0)

    X_torch = Var(X_rep[:(NUM_BATCHES*BATCH_SIZE),:].reshape([NUM_BATCHES,-1,d]))
    y_torch = Var(y_rep[:(NUM_BATCHES*BATCH_SIZE)].reshape([NUM_BATCHES,-1]))

    #Declare Network
    net = BayesianNetwork(inputSize = d,\
                        CLASSES = CLASSES, \
                        layers=np.array([16,16]), \
                        activations = np.array(['relu','relu','none']), \
                        SAMPLES = SAMPLES, \
                        BATCH_SIZE = BATCH_SIZE,\
                        NUM_BATCHES = NUM_BATCHES,\
                        hasScalarMixturePrior = True,\
                        PI = PI,\
                        SIGMA_1 = SIGMA_1,\
                        SIGMA_2 = SIGMA_2,\
                        GOOGLE_INIT= False).to(DEVICE)

    #Declare the optimizer
    optimizer = optim.SGD(net.parameters(),lr=1e-4,momentum=0.95)

    for epoch in range(TRAIN_EPOCHS):
        train(net, optimizer,data=X_torch,target=y_torch,NUM_BATCHES=NUM_BATCHES)

    return net

def predict_NN(net,X_test,stage):
    TEST_SAMPLES = 10000
    CLASSES = 1
    BATCH_SIZE = 100
    d = X_test.shape[1]
    X_test_torch = Var(X_test)

    for i in range(len(net.layers)):
        if i==0:
            weight_mu_ls = net.layers[i].weight_mu.cpu().detach().numpy().flatten()
            weight_sigma_ls = net.layers[i].weight_sigma.cpu().detach().numpy().flatten()
            bias_mu_ls = net.layers[i].bias_mu.cpu().detach().numpy().flatten()
            bias_sigma_ls = net.layers[i].bias_sigma.cpu().detach().numpy().flatten()
        else:
            weight_mu_ls = np.concatenate([weight_mu_ls,net.layers[i].weight_mu.cpu().detach().numpy().flatten()])
            weight_sigma_ls = np.concatenate([weight_sigma_ls,net.layers[i].weight_sigma.cpu().detach().numpy().flatten()])
            bias_mu_ls = np.concatenate([bias_mu_ls,net.layers[i].bias_mu.cpu().detach().numpy().flatten()])
            bias_sigma_ls = np.concatenate([bias_sigma_ls,net.layers[i].bias_sigma.cpu().detach().numpy().flatten()])
    mu_posterior = np.concatenate([weight_mu_ls, bias_mu_ls])
    sigma_posterior = np.concatenate([weight_sigma_ls, bias_sigma_ls])

    output_arr = []
    param_arr = []
    outputs = torch.zeros(TEST_SAMPLES, X_test_torch.shape[0], CLASSES).to(DEVICE)
    for k in range(TEST_SAMPLES):
        outputs[k] = net.forward(X_test_torch)
        output_arr.append(outputs[k].cpu().detach().numpy().flatten())
        for i in range(len(net.layers)):
            if i==0:
                weight_ls = net.layers[i].weight.cpu().detach().numpy().flatten()
                bias_ls = net.layers[i].bias.cpu().detach().numpy().flatten()
            else:
                weight_ls = np.concatenate([weight_ls,net.layers[i].weight.cpu().detach().numpy().flatten()])
                bias_ls =  np.concatenate([bias_ls,net.layers[i].bias.cpu().detach().numpy().flatten()])   
        param_ls =  np.concatenate([weight_ls,bias_ls])
        param_arr.append(param_ls)

    output_arr = np.array(output_arr)
    param_arr = np.array(param_arr)

    cov = np.diag(sigma_posterior**2) 
    if stage == 1:
        critical_value = scipy.stats.chi2.ppf(1-(1-(1-(1-0.95)/2))/2, cov.shape[0])
    else:
        critical_value = scipy.stats.chi2.ppf(np.sqrt(1-(1-0.95)/2), cov.shape[0])
    v = param_arr - mu_posterior
    test_stat = np.diag(np.matmul(np.matmul(v,inv(cov)),v.T))
    accept = test_stat < critical_value
    pred_uniform = np.min(output_arr[accept,:],0)

    pred_mean = outputs.mean(0).data.cpu().numpy().squeeze(1) #Compute mean prediction
    pred_std = outputs.std(0).data.cpu().numpy().squeeze(1) #Compute standard deviation of prediction for each data point
    if stage == 1:
        outputs_arr = outputs.data.cpu().numpy().squeeze(axis=2)
        pred_quantile = np.quantile(outputs_arr,q = (1-(1-(1-0.975)/2))/2,axis = 0) #Compute the quantile
    else:
        outputs_arr = outputs.data.cpu().numpy().squeeze(axis=2)
        pred_quantile = np.quantile(outputs_arr,q = 1-np.sqrt(1-(1-0.975)/2),axis = 0) #Compute the quantile
    return pred_mean, pred_std, pred_quantile, pred_uniform

def sim_mab(seed,N,s_dim1,s_dim2,percent,noise,n_eval,W1_ls):
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    select_stage1 = [[0,1],[2,3]]
    select_stage2 = [[0,2],[1,3]]
    net2_ls = []
    for select in select_stage2:
        i1, i2 = select
        A1 = np.concatenate([a1_ls[i1],a1_ls[i2]]) 
        S1 = np.concatenate([S1_ls[i1],S1_ls[i2]]) 
        S2 = np.concatenate([S2_ls[i1],S2_ls[i2]]) 
        S2_input = np.concatenate([A1,S1,S2],axis=1)
        r = np.concatenate([reward_ls[i1],reward_ls[i2]]) 
        net2 = Bayesian_NN_est(S2_input,r,N)
        net2_ls.append(net2)

    NN1_non_pessi_ls = []
    NN1_pessi_ls = []
    NN1_quantile_ls = []
    NN1_uniform_ls = []
    for select in select_stage1:
        i1, i2 = select
        A1 = np.concatenate([a1_ls[i1],a1_ls[i2]]) 
        S1 = np.concatenate([S1_ls[i1],S1_ls[i2]]) 
        S2 = np.concatenate([S2_ls[i1],S2_ls[i2]])   
        S2_input = np.concatenate([A1,S1,S2],axis=1)

        est_mc_non_pessi = []
        est_mc_pessi = []
        est_mc_quantile = []
        est_mc_uniform = []
        for i in range(2):
            MC_mean, MC_std, MC_quantile, est_uniform_i=predict_NN(net2_ls[i],S2_input,2)       

            est_mc_non_pessi_i = MC_mean
            est_mc_pessi_i = MC_mean - scipy.stats.norm.ppf(np.sqrt(1-(1-0.975)/2))*MC_std
            est_mc_quantile_i = MC_quantile  

            est_mc_non_pessi.append(est_mc_non_pessi_i)
            est_mc_pessi.append(est_mc_pessi_i)
            est_mc_quantile.append(est_mc_quantile_i)
            est_mc_uniform.append(est_uniform_i)

        reward_non_pessi = np.max(est_mc_non_pessi,0)
        reward_pessi = np.max(est_mc_pessi,0)
        reward_quantile = np.max(est_mc_quantile,0)
        reward_uniform = np.max(est_mc_uniform,0)

        NN1_non_pessi = Bayesian_NN_est(S1,reward_non_pessi,N)
        NN1_non_pessi_ls.append(NN1_non_pessi)

        NN1_pessi = Bayesian_NN_est(S1,reward_pessi,N) 
        NN1_pessi_ls.append(NN1_pessi)

        NN1_quantile = Bayesian_NN_est(S1,reward_quantile,N)  
        NN1_quantile_ls.append(NN1_quantile)

        NN1_uniform = Bayesian_NN_est(S1,reward_uniform,N)  
        NN1_uniform_ls.append(NN1_uniform)

    est_mc_non_pessi = []
    est_mc_pessi = []
    est_mc_quantile = []
    est_mc_uniform = []
    for i in range(2):
        MC_mean, _, _, _=predict_NN(NN1_non_pessi_ls[i],S1_test,1)       
        est_mc_non_pessi.append(MC_mean)
        MC_mean, MC_std, _, _=predict_NN(NN1_pessi_ls[i],S1_test,1) 
        est_mc_pessi.append(MC_mean - scipy.stats.norm.ppf(1-(1-(1-(1-0.975)/2))/2)*MC_std)
        _, _, MC_quantile, _=predict_NN(NN1_quantile_ls[i],S1_test,1) 
        est_mc_quantile.append(MC_quantile)
        _, _ , _, est_uniform_i=predict_NN(NN1_uniform_ls[i],S1_test,1) 
        est_mc_uniform.append(est_uniform_i)

    arm1_non_pessi = np.argmax(est_mc_non_pessi,0)
    arm1_pessi = np.argmax(est_mc_pessi,0)
    arm1_quantile = np.argmax(est_mc_quantile,0)
    arm1_uniform = np.argmax(est_mc_uniform,0)

    S2_test_non_pessi = np.array([np.matmul(S1_test[i].reshape([1,-1]),W1_ls[arm1_non_pessi[i]]).flatten() for i in range(n_eval)])
    S2_test_pessi = np.array([np.matmul(S1_test[i].reshape([1,-1]),W1_ls[arm1_pessi[i]]).flatten() for i in range(n_eval)])
    S2_test_quantile = np.array([np.matmul(S1_test[i].reshape([1,-1]),W1_ls[arm1_quantile[i]]).flatten() for i in range(n_eval)])
    S2_test_uniform = np.array([np.matmul(S1_test[i].reshape([1,-1]),W1_ls[arm1_uniform[i]]).flatten() for i in range(n_eval)])

    est_mc_non_pessi = []
    est_mc_pessi = []
    est_mc_quantile = []
    est_mc_uniform = []
    for i in range(2):
        S2_input = np.concatenate([arm1_non_pessi.reshape([-1,1]),S1_test,S2_test_non_pessi],axis=1)
        MC_mean, _, _, _=predict_NN(net2_ls[i],S2_input,2)       
        est_mc_non_pessi.append(MC_mean)
        
        S2_input = np.concatenate([arm1_pessi.reshape([-1,1]),S1_test,S2_test_pessi],axis=1)
        MC_mean, MC_std, _, _=predict_NN(net2_ls[i],S2_input,2) 
        est_mc_pessi.append(MC_mean - scipy.stats.norm.ppf(np.sqrt(1-(1-0.975)/2))*MC_std)
        
        S2_input = np.concatenate([arm1_quantile.reshape([-1,1]),S1_test,S2_test_quantile],axis=1)
        _, _, MC_quantile, _=predict_NN(net2_ls[i],S2_input,2) 
        est_mc_quantile.append(MC_quantile)
        
        S2_input = np.concatenate([arm1_uniform.reshape([-1,1]),S1_test,S2_test_uniform],axis=1)
        _, _ , _, est_uniform_i=predict_NN(net2_ls[i],S2_input,2) 
        est_mc_uniform.append(est_uniform_i)

    arm2_non_pessi = np.argmax(est_mc_non_pessi,0)
    arm2_pessi = np.argmax(est_mc_pessi,0)
    arm2_quantile = np.argmax(est_mc_quantile,0)
    arm2_uniform = np.argmax(est_mc_uniform,0)

    reward_non_pessi = np.array([funct_ls[arm2_non_pessi[i]](S2_test_non_pessi[i]).flatten() for i in range(n_eval)]).flatten()
    reward_pessi = np.array([funct_ls[arm2_pessi[i]](S2_test_pessi[i]).flatten() for i in range(n_eval)]).flatten()
    reward_quantile = np.array([funct_ls[arm2_quantile[i]](S2_test_quantile[i]).flatten() for i in range(n_eval)]).flatten()
    reward_uniform = np.array([funct_ls[arm2_uniform[i]](S2_test_uniform[i]).flatten() for i in range(n_eval)]).flatten()

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

    return regret_mc_non_pessi, regret_mc_pessi, regret_mc_quantile, regret_mc_uniform

def generate_result(N,s_dim1,s_dim2,percent,noise,n_eval,W1_ls):
    regret_non_pessi_ls = []
    regret_pessi_ls = []
    regret_quantile_ls = [] 
    regret_uniform_ls = [] 

    for seed in tqdm(range(50)):
        regret_non_pessi, regret_pessi, regret_quantile, regret_uniform = sim_mab(seed,N,s_dim1,s_dim2,percent,noise,n_eval,W1_ls)

        regret_non_pessi_ls.append(regret_non_pessi)
        regret_pessi_ls.append(regret_pessi)
        regret_quantile_ls.append(regret_quantile)
        regret_uniform_ls.append(regret_uniform)

    out = [regret_non_pessi_ls, regret_pessi_ls, regret_quantile_ls, regret_uniform_ls]
    return out

string = "bnn_nonlinear_stage1"

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
    percent = 0.7
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