# Machine-Learning
(Samples and Projects)
# Gradient Descent Implementation  

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen) ![License](https://img.shields.io/badge/License-MIT-blue)  

This repository contains implementations of gradient descent to solve a simple quadratic function and to fit a linear model to a synthetic wave dataset. The project demonstrates the core concepts of gradient descent, including how to optimize parameter values to minimize a cost function effectively.  

## Table of Contents  
- [Overview](#overview)  
- [1. Solving f(x) = (x + 3)^2](#1-solving-fx--x--32)  
- [2. Fitting a Linear Model to a Wave Dataset](#2-fitting-a-linear-model-to-a-wave-dataset)  
- [Installation](#installation)  
- [Results](#results)  
- [License](#license)  

## Overview  
This project showcases two key applications of gradient descent in machine learning:  
1. Finding the minimum of a quadratic function.  
2. Fitting a linear regression model to a synthetic dataset.

## 1. Solving f(x) = (x + 3)^2  
This section implements gradient descent to find the minimum of the function \( f(x) = (x + 3)^2 \).  

- The gradient of the function is calculated as \( \text{grad}(f) = 2x + 6 \).  
- The process stops when the change in \( x \) is less than a defined precision.  

**Python Code:**  
```python  
x0 = 0  
learning_rate = 0.01  
precision = 0.00001  
max_iter = 1000  

# Gradient (derivative) function  
grad = lambda x: 2 * x + 6  

for i in range(max_iter):  
    x1 = x0 - learning_rate * grad(x0)  
    dx = abs(x1 - x0)  
    if dx > precision:  
        x0 = x1  
    else:  
        break  

print(f'Reach x = {x1:.3f} with last dx = {dx:.5f} after {i} iterates')
```
## 2. Fitting a Linear Model to a Wave Dataset
This section demonstrates how to fit a linear regression model to a generated wave dataset using gradient descent.

Dataset: The dataset is created using mglearn.datasets.make_wave, consisting of 100 samples.
Model: The model parameters are optimized using the gradient descent algorithm.
Python Code:

```python
import numpy as np  
import matplotlib.pyplot as plt  
import mglearn  
from sklearn.model_selection import train_test_split  

# Generate synthetic wave dataset  
X, y = mglearn.datasets.make_wave(n_samples=100)  

# Split dataset into training and test sets  
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)  

# Reshape the input data  
x_train = x_train.reshape(-1, 1)  
y_train = y_train.reshape(-1, 1)  

# Add bias (intercept) term  
intercept = np.ones_like(x_train).reshape(-1, 1)  
X_bias = np.c_[intercept, x_train]  # Add bias as the first column  

# Gradient Descent Algorithm  
learning_rate = 0.1  
max_iter = 1000  
theta = np.zeros((X_bias.shape[1], 1))  # Initialize weights  
cost_function = []  

for i in range(max_iter):  
    y_hat = np.dot(X_bias, theta)           
    error = y_hat - y_train                  
    gradient = np.dot(X_bias.T, error) / x_train.shape[0]  
    theta -= learning_rate * gradient         
    cost = (error ** 2).mean()               
    cost_function.append(cost)  
Plotting Results
The fitting results and the cost function over iterations are visualized.

Cost Function Visualization:

python
plt.plot(cost_function)  
plt.title('Cost Function (Mean Squared Error) over Iterations')  
plt.xlabel('Iterations')  
plt.ylabel('Mean Squared Error')  
plt.show()
```
## Installation
To run this project, ensure you have Python 3.x installed along with the required libraries. You can install the libraries using pip:

```bash
pip install numpy matplotlib mglearn scikit-learn
```  
## Usage
Clone the repository:
``` bash
git clone https://github.com/mhkhodadadi/Machine-Learning.git  
cd Machine-Learning
``` 
Run the script containing the gradient descent implementations:
```bash
python your_script_name.py 
``` 
## Results
The implementation should yield:

The optimized value of ( x ) for the quadratic function.
A visualization of the linear model fitted to the synthetic wave dataset, along with the progression of the cost function over iterations.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
