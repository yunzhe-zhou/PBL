# Testing Directed Acyclic Graph via Structural, Supervised and Generative Adversarial Learning

This repository contains the implementation for the paper ["Testing Directed Acyclic Graph via Structural, Supervised and Generative Adversarial Learning
"](https://arxiv.org/pdf/2106.01474.pdf) in Python. 

## Summary of the paper

In this article, we propose a new hypothesis testing method for directed acyclic graph
(DAG). While there is a rich class of DAG estimation methods, there is a relative paucity
of DAG inference solutions. Moreover, the existing methods often impose some specific
model structures such as linear models or additive models, and assume independent data
observations. Our proposed test instead allows the associations among the random vari-
ables to be nonlinear and the data to be time-dependent. We build the test based on some
highly flexible neural networks learners. We establish the asymptotic guarantees of the
test, while allowing either the number of subjects or the number of time points for each
subject to diverge to infinity. We demonstrate the efficacy of the test through simulations
and a brain connectivity network analysis.


**Figures**:  

| Size | Power | Power Difference |
| :-------:    |  :-------: |  :-------: |
| <img align="center" src="sim_null.png" alt="drawing" width="500">   | <img align="center" src="sim_alter.png" alt="drawing" width="500" >  | <img align="center" src="sim_diff.png" alt="drawing" width="500" >  

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
