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

def Bayesian_NN(X,y,X_test,N,n_eval):
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
    X_test_torch = Var(X_test)
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
    outputs = torch.zeros(TEST_SAMPLES, n_eval, CLASSES).to(DEVICE)
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

    cov = sigma_posterior**2
    cov_inv = 1/(sigma_posterior**2)
    critical_value = scipy.stats.chi2.ppf(np.sqrt(0.95), cov.shape[0])
    v = param_arr - mu_posterior
    test_stat = np.sum(v**2 * cov_inv.reshape([1,-1]),1)
    accept = test_stat < critical_value
    pred_uniform = np.min(output_arr[accept,:],0)

    pred_mean = outputs.mean(0).data.cpu().numpy().squeeze(1) #Compute mean prediction
    pred_std = outputs.std(0).data.cpu().numpy().squeeze(1) #Compute standard deviation of prediction for each data point
    outputs_arr = outputs.data.cpu().numpy().squeeze(axis=2)
    pred_quantile = np.quantile(outputs_arr,q = 1-np.sqrt(0.975),axis = 0) #Compute the quantile

    return pred_mean, pred_std, pred_quantile, pred_uniform

def sim_mab(seed,n_arms,w_dim,N,percent,n_eval,noise):
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    est_mc_non_pessi = []
    est_mc_pessi = []
    est_mc_quantile = []
    est_mc_uniform = []
    for i in range(n_arms):
        X_train = X_ls[i]
        funct = funct_ls[i]
        y_train = funct(X_train) + np.random.randn(X_train.shape[0]) * noise

        MC_mean, MC_std, MC_quantile, MC_uniform = Bayesian_NN(X_train,y_train,X_test,N,n_eval)

        est_mc_non_pessi_i = MC_mean
        est_mc_pessi_i = MC_mean - scipy.stats.norm.ppf(np.sqrt(0.975))*MC_std
        est_mc_quantile_i = MC_quantile  
        est_mc_uniform_i = MC_uniform
        est_mc_non_pessi.append(est_mc_non_pessi_i)
        est_mc_pessi.append(est_mc_pessi_i)
        est_mc_quantile.append(est_mc_quantile_i)
        est_mc_uniform.append(est_mc_uniform_i)

    arm_non_pessi = np.argmax(est_mc_non_pessi,0)
    arm_pessi = np.argmax(est_mc_pessi,0)
    arm_quantile = np.argmax(est_mc_quantile,0)
    arm_uniform = np.argmax(est_mc_uniform,0)

    eval_mc_non_pessi = []
    eval_mc_pessi = []
    eval_mc_quantile = []
    eval_mc_uniform = []
    eval_optimal = []
    for i in range(n_eval):
        np.random.seed(seed*1000 + i)
        eval_mc_non_pessi.append(funct_ls[arm_non_pessi[i]](X_test[i]))
        np.random.seed(seed*1000 + i)
        eval_mc_pessi.append(funct_ls[arm_pessi[i]](X_test[i]))
        np.random.seed(seed*1000 + i)
        eval_mc_quantile.append(funct_ls[arm_quantile[i]](X_test[i]))
        np.random.seed(seed*1000 + i)
        eval_mc_uniform.append(funct_ls[arm_uniform[i]](X_test[i]))
        np.random.seed(seed*1000 + i)
        eval_optimal.append(np.max([funct(X_test[i]) for funct in funct_ls]))
    regret_mc_non_pessi = np.sum(eval_optimal) -  np.sum(eval_mc_non_pessi)
    regret_mc_pessi = np.sum(eval_optimal) -  np.sum(eval_mc_pessi)
    regret_mc_quantile = np.sum(eval_optimal) -  np.sum(eval_mc_quantile)
    regret_mc_uniform = np.sum(eval_optimal) -  np.sum(eval_mc_uniform)
    return regret_mc_non_pessi, regret_mc_pessi, regret_mc_quantile, regret_mc_uniform

def generate_result(n_arms,w_dim,N,percent,n_eval,noise):
    regret_non_pessi_ls = []
    regret_pessi_ls = []
    regret_quantile_ls = [] 
    regret_uniform_ls = [] 

    for seed in tqdm(range(50)):
        regret_non_pessi, regret_pessi, regret_quantile, regret_uniform = sim_mab(seed,n_arms,w_dim,N,percent,n_eval,noise)

        regret_non_pessi_ls.append(regret_non_pessi)
        regret_pessi_ls.append(regret_pessi)
        regret_quantile_ls.append(regret_quantile)
        regret_uniform_ls.append(regret_uniform)

    out = [regret_non_pessi_ls, regret_pessi_ls, regret_quantile_ls, regret_uniform_ls]
    return out

string = "bnn_linear_stage1"

info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.95
    noise = 0.1
    n_eval = 500
    w_dim = 3
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.85
    noise = 0.1
    n_eval = 500
    w_dim = 3
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.75
    noise = 0.1
    n_eval = 500
    w_dim = 3
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent),N_arr)
    
info_ls = []
N_arr = []
for t in range(6):
    N = 500*(t+1)
    n_arms = 2
    percent = 0.5
    noise = 0.1
    n_eval = 500
    w_dim = 3
    info = generate_result(n_arms,w_dim,N,percent,n_eval,noise)
    N_arr.append(N)
    info_ls.append(info)
    np.save("../data/info_ls_"+string+"_"+str(percent),info_ls)
    np.save("../data/N_arr_"+string+"_"+str(percent),N_arr)