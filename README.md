# Optimizing Pessimism in Dynamic Treatment Regimes: A Bayesian Learning Approach

This repository contains the implementation for the paper ["Optimizing Pessimism in Dynamic Treatment Regimes: A Bayesian Learning Approach"](https://openreview.net/pdf?id=8tABKfhBBpJ) in Python. 

## Summary of the paper

In this article, we propose a novel pessimism-based Bayesian learning method for optimal dynamic treatment regimes in the offline setting. When the coverage condition does not hold, which is common for offline data, the existing solutions would produce sub-optimal policies. The pessimism principle addresses this issue by discouraging recommendation of actions that are less explored conditioning on the state. However, nearly all pessimism-based methods rely on a key hyper-parameter that quantifies the degree of pessimism, and the performance of the methods can be highly sensitive to the choice of this parameter. We propose to integrate the pessimism principle with Thompson sampling and Bayesian machine learning for optimizing the degree of pessimism. We derive a credible set whose boundary uniformly lower bounds the optimal Q-function, and thus no longer require the tuning of the degree of pessimism. We develop a general Bayesian learning method that works with a range of models, from Bayesian linear basis model to Bayesian neural network model. We develop the computational algorithm based on variational inference, which is highly efficient and scalable. We establish the theoretical guarantees of the proposed method, and show empirically that it outperforms the existing state-of-the-art solutions through both simulations and a real data example. 


**Figures**:  

| Linear | Nonlinear | 
| :-------:    |  :-------: | 
| <img align="center" src="stage1_linear.png" alt="drawing" width="500">   | <img align="center" src="stage1_nonlinear.png" alt="drawing" width="500" > |

## Requirement

+ Python 3.6
    + numpy 1.18.5
    + scipy 1.5.4
    + torch 1.0.0
    + tensorflow 2.1.3
    + sklearn 0.23.2

+ R 3.6.3
    + clrdag (https://github.com/chunlinli/clrdag)


## File Overview
- `src/`: This folder contains all python codes used in numerical experiments and real data analysis.
  - 'inference.py' is used for data generating and supervised and generative adversarial learning.
  - `infer_utils.py` contains the utility functions to implement hypothesis testing.
  - `main.py` is an entrance to be used in command line. We can type `python main.py` to reproduce all the results of this paper.
  - `main_lrt.R` is to implement the methods in ["Likelihood ratio tests for a large directed acyclic graph"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7508303/)
  - `nonlinear_learning.py` is used for structural learning of the graphs. (Refers to https://github.com/xunzheng/notears)
  - `plot.py` contains the functions to load test results and draw plots.
- `data/`: This folder where the output results and the dataset should be put.
  - 'data_process.R' is used for preprocessing the HCP dataset. 
