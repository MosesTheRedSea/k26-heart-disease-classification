import copy, math
import numpy as np
from tqdm.auto import tqdm

def compute_cost(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    A = sigmoid(z)
    epsilon = 1e-15  
    cost = -(1/m) * np.sum(
        y * np.log(A + epsilon) +
        (1 - y) * np.log(1 - A + epsilon)
    )
    lambda_ =  0.5
    cost += (lambda_ / (2*m)) * np.sum(w**2)
    return cost

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    A = sigmoid(z)
    error = A - y
    dj_dw = (1/m) * np.dot(X.T, error)
    dj_db = (1/m) * np.sum(error)
    lambda_ =  0.5
    dj_dw += (lambda_/m) * w
    return dj_db, dj_dw

def gradient_descent(X, y, w, b, alpha, num_iters):
    J_history = []
    for i in tqdm(range(num_iters), desc="Training Progress"):
        dj_db, dj_dw = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(compute_cost(X, y, w, b))
    return w, b, J_history

def predict_probabilities(w, b, X):
    return sigmoid(np.dot(X, w) + b)