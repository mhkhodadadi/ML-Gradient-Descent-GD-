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

# Understanding the Gradient of Matrix in Linear Regression  

## Introduction  

In linear regression and other machine learning algorithms, understanding the gradient is fundamental for optimizing model parameters (weights). This document explains the mathematical basics of the gradient, particularly focusing on the use of the input feature matrix \(X\) and error vector in its computation.  

## 1. What is a Gradient?  

The gradient of a loss function \(L(w)\), where \(w\) are model parameters, is defined as:  

\[  
\nabla L(w) = \left[ \frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \ldots, \frac{\partial L}{\partial w_n} \right]  
\]  

The gradient vector indicates the steepest direction of increase for the loss function. To minimize the loss, we update model parameters in the opposite direction of the gradient.  

## 2. Gradient for Linear Regression  

The predictions in linear regression are expressed as:  

\[  
\hat{y} = X \cdot w  
\]  

Where:  
- \(X\) is the input feature matrix of shape \(m \times n\) (with \(m\) samples and \(n\) features).  
- \(w\) is the weight vector of shape \(n \times 1\).  
- \(\hat{y}\) is the predicted output of shape \(m \times 1\).  

The Mean Squared Error (MSE) loss function is commonly used:  

\[  
L(w) = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2  
\]  

Where \(y_i\) is the actual target value.  

## 3. Deriving the Gradient  

To compute the gradient, we derive how the loss changes with respect to weights \(w\):  

1. **Calculate the Error**:  

\[  
\text{error} = y - \hat{y} = y - (X \cdot w)  
\]  

2. **Gradient of the Loss**:  

We compute the derivative of the MSE loss with respect to the weights:  

\[  
\nabla L(w) = \frac{\partial L(w)}{\partial w} = \frac{\partial}{\partial w} \left( \frac{1}{m} \sum_{i=1}^{m} (y_i - (X \cdot w)_i)^2 \right)  
\]  

Using the chain rule, we arrive at:  

\[  
\nabla L(w) = \frac{-2}{m} X^T \cdot \text{error}  
\]  

Here, \(X^T\) (the transpose of \(X\)) allows multiplication with the \(m \times 1\) error vector, resulting in a gradient of shape \(n \times 1\) (matching the shape of \(w\)).  

## 4. Why Use \(X^T\) and the Error Vector?  

- **Using \(X^T\)**: The transpose operation changes the dimensions so we can compute each feature's contributions across all samples, aggregating the influence of each feature on the total loss.  
  
- **Error Vector**: It quantifies how predictions differ from actual values. Multiplying this error by \(X^T\) provides the total gradient contribution for each weight based on associated features.  

## Conclusion  

The gradient \(\nabla L(w)\) provides essential information for updating weights to minimize the loss function. Using \(X^T\) ensures that we correctly compute contributions from each feature, while the error vector informs us of discrepancies between predicted and actual values.

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
