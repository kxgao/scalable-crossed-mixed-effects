# scalable-crossed-mixed-effects

This repository contains Python code for scalable parameter estimation and inference in massive linear mixed models with crossed random effects. 
The algorithms are taken from our papers:
1. Katelyn Gao and Art Owen. Efficient moment calculations for variance components in large unbalanced crossed random effects models. Electronic Journal of Statistics, 11(1): 1235-1296, 2017. http://projecteuclid.org/euclid.ejs/1492135234.
2. Katelyn Gao and Art Owen. Estimation and inference for very large linear mixed effects models. Arxiv e-prints, 2016. http://arxiv.org/abs/1610.08088v2.

We assume that there are two crossed factors; the examples from e-commerce in the papers have as factors users and products.
The data are assumed to reside on one machine.

# Usage

For the crossed random effects model, download cre.py. For the linear mixed model, download mixed.py.

## cre.py

From the command line, run python cre.py fileName.txt.
1. fileName.txt contains the data in log-file format: (i,j,Yij), where i is the level of the first factor, j is the level of the second factor, and Yij is the response of interest.
2. Prints out estimated variance components and their conservative variances.

## mixed.py

From the command line, run python mixed.py fileName.txt
1. fileName.txt contains the data in log-file format: (i,j,Yij,xij), where i is the level of the first factor, j is the level of the second factor, Yij is the response of interest, and xij are the predictors. 
2. Prints out estimated regression coefficients and variance components and their asymptotic variances. 
