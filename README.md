# NTK on complex models
## EECS 6699: Final Project, Spring 2022

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)]()

This is the repository of source code for our final project Big of EECS E6699: Mathmatics of deep learning.

This repository contains codes for:


1. Investigating the convergence of NTK for GNN, RNN, Transformer
2. Evaluating the performance of NTK-related methods on downstream tasks for GNN, RNN, Transformer
3. Exploring the lazy training.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Convergence of NTK](#convergence-of-ntk)
- [Performance Evaluation](#performance-evaluation)
- [Lazy Training](#lazy-training)

## Background

**In this project, we try to explore characteristics of neural tangent kernel methods on following architecture**
>1. Graph Neural Network.
>2. Recurrent Neural Network.
>3. Transformer. 

**The whole project can be divided into three parts:**
>1. Investigating the convergence of NTK for GNN, RNN, Transformer
>2. Evaluating the performance of NTK-related methods on downstream tasks for GNN, RNN, Transformer
>3. Exploring the lazy training.


## Install
Set up the environment needed for this project. And then start jupyter notebook to see all the experiments.
```sh
$ pip install -r requirements.txt
$ jupyter notebook
```
You could find models we implement in [models](models) 

How to calculate theoretical NTK could be found in [kernels](kernels)

## Convergence of NTK

Code and result to show convergence for NTK could be found in following files:

1. GNN:  [gnn_ntk_convergence.ipynb](gnn_ntk_convergence.ipynb)
2. RNN: [rnn_ntk_convergence.ipynb](rnn_ntk_convergence.ipynb)
3. Transformer: [transformer_ntk_convergence.ipynb](transformer_ntk_convergence.ipynb)
4. All results in the same plot: [transformer_ntk_convergence.ipynb](transformer_ntk_convergence.ipynb)

## Performance Evaluation

Code and result to evaluate the performance of NTK-related methods on downstream tasks could be found in following files:

1. GNN:  [gnn_performance_evaluation.ipynb](gnn_performance_evaluation.ipynb)
2. RNN: [rnn_performance_evaluation.ipynb](rnn_performance_evaluation.ipynb)
3. Transformer: [transformer_performance_evaluation.ipynb](transformer_performance_evaluation.ipynb)

## Lazy Training

Code and result to explore lazy training could be found in: [gnn_lazy_training.ipynb](gnn_lazy_training.ipynb)






