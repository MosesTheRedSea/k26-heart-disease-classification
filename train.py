import os
import re
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import pandas as pd
from utils.lr_utils import *
from utils.visualize import *
from model.heart_disease import *

if __name__ == '__main__':

    X_train, y_train, test_X, test_ids = load_dataset('/home/hice1/madewolu9/scratch/madewolu9/Data/BC-heart-disease/train.csv', '/home/hice1/madewolu9/scratch/madewolu9/Data/BC-heart-disease/test.csv')

    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    X_train = (X_train - X_mean) / X_std
    test_X  = (test_X - X_mean) / X_std

    print("First five elements in X_train are:\n", X_train[:5])
    print("Type of X_train:",type(X_train))

    print("First five elements in y_train are:\n", y_train[:5])
    print("Type of y_train:",type(y_train))

    print ('The shape of X_train is: ' + str(X_train.shape))
    print ('The shape of y_train is: ' + str(y_train.shape))
    print ('We have m = %d training examples' % (len(y_train)))

    plot_and_save_data(X_train, y_train, base_dir="/home/hice1/madewolu9/scratch/madewolu9/heart-disease-prediction/outputs/train")

    initial_w = np.zeros(X_train.shape[1])
    initial_b = 0.0

    alpha = 0.01
    iterations = 2000
    
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, alpha, iterations)
    
    probabilities = predict_probabilities(w_final, b_final, test_X)

    # Final training loss
    final_loss = compute_cost(X_train, y_train, w_final, b_final)

    # Predictions
    train_probs = sigmoid(np.dot(X_train, w_final) + b_final)
    train_preds = (train_probs > 0.5).astype(int)

    # Accuracy
    train_accuracy = np.mean(train_preds == y_train)

    print(f"\nFinal Training Loss: {final_loss:.6f}")
    print(f"Final Training Accuracy: {train_accuracy * 100:.2f}%")

    results_base_dir = "/home/hice1/madewolu9/scratch/madewolu9/heart-disease-prediction/outputs/results"

    run_dir = save_run_outputs(results_base_dir)

    submission = pd.DataFrame({
        "id": test_ids,
        "Heart Disease": probabilities
    })

    submission_path = os.path.join(run_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")
