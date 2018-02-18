# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:29:53 2018

Module containing the functions for the tasks of exercise 1

@author: Karina Shyrokykh
"""
import numpy as np

# Warm-up exercise
def getIdentityMat(n):
    return np.identity(n)

# Calculate sum of residuals (cost function)
def computeCost(X, y, theta):
    m = len(y) # number of data points
    h = np.matmul(X, theta) # hypotesis for given value pair in X
    res = h.transpose() - y # residuals
    J = 1/(2*m) * np.sum(np.power(res, 2)) # contribution to cost function of given X and y
    return J

# Function for gradient descent 
def gradientDescent(X, y, theta, alpha, n_itrerations):
    m = len(y)
    J_hist = np.zeros((n_itrerations, 1)) #one-dimentional vector of zeros to save the cost J in every iteration
    for i in range(n_itrerations):
        h = np.matmul(X, theta) # hypotesis for given value pair in X
        res = h.transpose() - y # residuals
        dJ = 1/m * np.matmul(res, X) # contribution to cost function of given X and y
        theta = theta - alpha * dJ.transpose() # gradient update
        J_hist[i] = computeCost(X, y, theta)
    return (theta, J_hist)
        
        
        
    