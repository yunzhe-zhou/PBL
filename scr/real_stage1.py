string = "real_stage1"

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/sepsis_processed_state_action.csv')
file1 = open('data/state_features.txt', 'r')
Lines = file1.readlines()
state_names = [line.strip() for line in Lines]

time_id = np.array(df['bloc'])
id1 = np.where((time_id == 2)==True)[0] - 1
id2 = np.where((time_id == 2)==True)[0]

df_state = df[state_names]
df_action = df[['iv_input','vaso_input']]
df_reward = -df[['SOFA']]

s1 = df_state.to_numpy()[id1,:]
s2 = df_state.to_numpy()[id2,:]

a1 = df_action.to_numpy()[id1,:]
a2 = df_action.to_numpy()[id2,:]

r1 = df_reward.to_numpy().flatten()[id1]
r2 = df_reward.to_numpy().flatten()[id2]

s1 = normalize(s1, axis=0, norm='max')
s2 = normalize(s2, axis=0, norm='max')
r1 = (r1 - np.mean(r1))/np.std(r1)
r2 = (r2 - np.mean(r2))/np.std(r2)

s1_train,s1_test,a1_train,a1_test,r1_train,r1_test,s2_train,s2_test,a2_train,a2_test,r2_train,r2_test = train_test_split(s1,a1,r1,s2,a2,r2,test_size=0.5,random_state=1)


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
    TRAIN_EPOCHS = 100
    SAMPLES = 5
    TEST_SAMPLES = 1000
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
        print(epoch)

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
        if k%100 ==0:
            print(k)
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
    critical_value = scipy.stats.chi2.ppf((0.95)**(1/25), cov.shape[0])
    v = param_arr - mu_posterior
    test_stat = np.sum(v**2 * cov_inv.reshape([1,-1]),1)
    accept = test_stat < critical_value
    pred_uniform = np.min(output_arr[accept,:],0)

    pred_mean = outputs.mean(0).data.cpu().numpy().squeeze(1) #Compute mean prediction
    pred_std = outputs.std(0).data.cpu().numpy().squeeze(1) #Compute standard deviation of prediction for each data point
    outputs_arr = outputs.data.cpu().numpy().squeeze(axis=2)
    pred_quantile = np.quantile(outputs_arr,q = 1-(0.975)**(1/25),axis = 0) #Compute the quantile

    return pred_mean, pred_std, pred_quantile, pred_uniform


est_mc_non_pessi = []
est_mc_pessi = []
est_mc_quantile = []
est_mc_uniform = []
action_ls = []
for i in range(5):
    for j in range(5):
        arm_train = (a1_train[:,0] == i) * (a1_train[:,1] == j)
        X_train = s1_train[arm_train,:]
        y_train = r1_train[arm_train]
        X_test = s1_test
        N = X_train.shape[0]
        n_eval = s1_test.shape[0]
        torch.manual_seed(0)
        MC_mean, MC_std, MC_quantile, MC_uniform = Bayesian_NN(X_train,y_train,X_test,N,n_eval)  

        est_mc_non_pessi_i = MC_mean
        est_mc_pessi_i = MC_mean - scipy.stats.norm.ppf((0.975)**(1/25))*MC_std
        est_mc_quantile_i = MC_quantile  
        est_mc_uniform_i = MC_uniform

        est_mc_non_pessi.append(est_mc_non_pessi_i)
        est_mc_pessi.append(est_mc_pessi_i)
        est_mc_quantile.append(est_mc_quantile_i)
        est_mc_uniform.append(est_mc_uniform_i)

        action_ls.append([i,j])
        print([i,j])

arm_non_pessi = np.argmax(est_mc_non_pessi,0)
arm_pessi = np.argmax(est_mc_pessi,0)
arm_quantile = np.argmax(est_mc_quantile,0)
arm_uniform = np.argmax(est_mc_uniform,0)      

def generate_tab(arm_eval):
    act_tab = np.zeros([5,5])
    for arm in arm_eval:
        i,j = action_ls[arm]
        act_tab[i,j] += 1 
    return np.int64(act_tab)

act_tab = np.zeros([5,5])
for i in range(5):
    for j in range(5):
        act_tab[i,j] = np.sum((a1_test[:,0] == i) * (a1_test[:,1] == j))

        
np.save("action_"+string,act_tab)
np.save("non_pessi_"+string,generate_tab(arm_non_pessi))
np.save("pessi_"+string,generate_tab(arm_pessi))
np.save("quantile_"+string,generate_tab(arm_quantile))
np.save("uniform_"+string,generate_tab(arm_uniform))