import copy
import numpy as np
import math
from typing import Tuple

def r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R-squared value for linear regression.

    Parameters:
        y (np.ndarray[float, ndim=1]): The actual target values of shape (m,).
        y_pred (np.ndarray[float, ndim=1]): The predicted target values of shape (m,).

    Returns:
        float: The R-squared value.
    """
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

def mae(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean absolute error between the predicted and actual target values.

    Parameters:
        y (np.ndarray[float, ndim=1]): The actual target values of shape (m,).
        y_pred (np.ndarray[float, ndim=1]): The predicted target values of shape (m,).

    Returns:
        float: The mean absolute error.
    """
    return np.mean(np.abs(np.subtract(y, y_pred)))

def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean squared error between the predicted and actual target values.

    Parameters:
        y (np.ndarray[float, ndim=1]): The actual target values of shape (m,).
        y_pred (np.ndarray[float, ndim=1]): The predicted target values of shape (m,).

    Returns:
        float: The mean squared error.
    """
    return np.mean(np.square(np.subtract(y, y_pred)))

def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the root mean squared error between the predicted and actual target values.

    Parameters:
        y (np.ndarray[float, ndim=1]): The actual target values of shape (m,).
        y_pred (np.ndarray[float, ndim=1]): The predicted target values of shape (m,).

    Returns:
        float: The root mean squared error.
    """
    return np.sqrt(mse(y, y_pred))
def compute_cost_matrix(X : np.ndarray, y: np.ndarray, w: np.ndarray, b: float, verbose=False) -> float:
    '''
    Compute cost for linear regression
    Args:
        X (np.ndarray[float, ndim=2]): The input features of shape (m, n) where m is the number of samples and n is the number of features.
        y (np.ndarray[float, ndim=1]): The target values of shape (m,).
        w (np.ndarray[float, ndim=1]): The coefficients of shape (n,).
        b (float): The intercept.
        verbose (bool): If true, print the cost every 1000 iterations.
        
    Returns:
        total_cost (float): The total cost.
    '''

    predictions = X @ w + b
    total_cost = (1/ 2) * mse(y, predictions)

    if verbose: print("f_wb:")
    if verbose: print(predictions)

    return total_cost

def compute_gradient_matrix(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
    """
    Compute the gradient matrix for linear regression.

    Parameters:
        X (np.ndarray[float, ndim=2]): The input features of shape (m, n) where m is the number of samples and n is the number of features.
        y (np.ndarray[float, ndim=1]): The target values of shape (m,).
        w (np.ndarray[float, ndim=1]): The coefficients of shape (n,).
        X (np.ndarray): The input features of shape (m, n) where m is the number of samples and n is the number of features.
        y (np.ndarray): The target values of shape (m,).
        w (np.ndarray): The coefficients of shape (n,).
        b (float): The bias term.

    Returns:
        Tuple[np.ndarray[float, ndim=1], float]: The gradient of the weights and the gradient of the bias.
        Tuple[np.ndarray, float]: The gradient of the weights and the gradient of the bias.
    """
   
    m, _ = X.shape
    predictions: np.ndarray = X @ w + b
    error: np.ndarray = np.subtract(predictions, y)
    gradient_weights: np.ndarray = (1/m) * (X.T @ error)
    gradient_bias: float = (1/m) * np.sum(error)

    return  gradient_weights, gradient_bias



def batch_gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function ): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X : (array_like Shape (m,n)    matrix of examples 
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m, _ = X.shape
    
    # An array to store values at each iteration primarily for graphing later
    hist = {"cost": [], "params": [], "grads": [], "iter": []}
    
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    save_interval = np.ceil(num_iters/10000) # prevent resource exhaustion for long runs


    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        weights, bias = gradient_function(X, y, w, b)
        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * weights               
        b = b - alpha * bias   
        # print("Running gradient descent...")            
      
        # Save cost J,w,b at each save interval for graphing
        if i == 0 or i % save_interval == 0:     
            hist["cost"].append(cost_function(X, y, w, b))
            hist["params"].append([w,b])
            hist["grads"].append([weights,bias])
            hist["iter"].append(i)

    return w, b, hist #return w,b and history for graphing
