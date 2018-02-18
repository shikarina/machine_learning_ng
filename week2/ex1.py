# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:40:55 2018

@author: Karina Shyrokykh
"""

# Load necessary packages
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cmap as cm
#import seaborn as sns
import os
#from mpl_toolkits.mplot3d import Axes3D



# Clear workspace
clear = lambda: os.system('cls')
clear()

## Set wd
os.chdir('...\\week2')
os.getcwd()

## 1. Warm-up exercise =============================
# link the module containing the defined function
import ex1funcs as e1

# Compute the identity matrix
n = 5
I_n = e1.getIdentityMat(n)
# Print te result
print('Identity matrix of size ', n, ':\n', I_n)

## 1. Plotting the data =============================
# Load the data
data = np.loadtxt('data\\ex1data1.txt', delimiter=',')
print(data)
x = data[:, 0]
y = data[:, 1]
m = len(y) # number of entries

# Plot the data
plt.scatter(x, y, c="r", alpha=0.5, marker='x',
            label="Available data")
plt.xlabel("City population [x 10^4]")
plt.ylabel("Profit [x $10^4]")
plt.legend(loc=2)

## 3. Fit the regression line =============================
x_0 = np.ones(m) # generate a column of ones
X   = np.column_stack((x_0, x)) # combine two colomns

n_itrerations = 1500 # number of iterations
alpha = 0.01 # learning rate
theta = np.zeros((2, 1)) # initial teta_0 and theta_1

# Calculate cost function
J = e1.computeCost(X, y, theta)
print('Cost: ', J, ' vs. expected 32.07') # initial cost: 32.07 (exactly as expected)

# Further theta-vector to check cost function
theta_prime = np.array([[-1], [2]])
J = e1.computeCost(X, y, theta_prime)
print('Cost: ', J, ' vs. expected 54.24') # 54.24 is the correct answer

# Run the gradient-descent algorithm to minimize the cost function
theta_star, J_hist = e1.gradientDescent(X, y, theta, alpha, n_itrerations)
print('Optimized theta: ', theta_star.transpose() , ' vs. expected [[-3.63  1.67]] ') # 54.24 correct answer

# Add the obtained fitted line to the plot
y_fitted = theta_star[0] + theta_star[1]*x
plt.plot(x, y_fitted, c="g", alpha=0.5,
            label="Fitted line")
plt.show() # show the plot with both data values and fitted line

# Visuzlize the algorithm convergence
plt.plot(J_hist, c="b", alpha=0.5,
            label="Algorithm convergence")
plt.xlabel('Iiteration')
plt.ylabel('Cost function')
plt.legend(loc=0)
plt.show()
# The cost does decrease with iterations => step size alpha is chosen properly!

# Predict the profit for cities with population 35,000 and 70,000
x_1 = 3.5
y_1 = theta_star[0] + theta_star[1]*x_1
print('For a city with', x_1, 'citizens, the predicted profit is', y_1*10000)
x_2 = 7
y_2 = theta_star[0] + theta_star[1]*x_2
print('For a city with', x_2, 'citizens, the predicted profit is', y_2*10000)


## 4. Visuzlize the cost function =============================
# Create a grid of coordinates for 3D plotting
theta0_values = np.linspace(-10, 10, 100) # 100 vlues on interval [-10, 10]
theta1_values = np.linspace(-1, 4, 100)   # 100 vlues on interval [-1, 4]
theta0_grid, theta1_grid = np.meshgrid(theta0_values, theta1_values, indexing='xy') # grid of coords
J = np.zeros((theta0_values.size, theta1_values.size)) # initial values for z coordinate (cost values)

# Calculate the values of the cost function for all grid points
for (i, j), v in np.ndenumerate(J):
    J[i, j] = e1.computeCost(X, y, theta=[[theta0_grid[i, j]], [theta1_grid[i, j]]])

# Plot the cost values
fig = plt.figure(figsize=(8, 4)) # empty figure
# Contour plot
CS = plt.contour(theta0_grid, theta1_grid, J, 
                 np.logspace(-2, 3, 20),
                 cmap=plt.cm.jet)
plt.clabel(CS, fontsize=9, inline=1)
plt.scatter(theta_star[0], theta_star[1], c='r', marker='x')
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$',
                  fontsize=17)
    ax.set_ylabel(r'$\theta_1$',
                  fontsize=17)
# 3D visualization
fig = plt.figure(figsize=(8, 4)) # empty figure
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_grid, theta1_grid, J, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax.set_zlabel('Cost')
ax.set_zlim(J.min(), J.max())
ax.view_init(elev=15, azim=230)
ax.set_xlabel(r'$\theta_0$',
              fontsize=17)
ax.set_ylabel(r'$\theta_1$',
              fontsize=17)
