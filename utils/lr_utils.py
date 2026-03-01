import os
import csv
import h5py
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from tqdm.auto import tqdm

def parse_train_dataset(directory):
    X, Y = [], []
    with open(directory, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in csv_reader:
            if count != 0:
                curr_data =  np.array(row[:14], dtype=float)
                X.append(curr_data)
                curr_label = np.array(1 if row[14] == "Presence" else 0, dtype=int)
                Y.append(curr_label)
            count += 1
    return np.array(X), np.array(Y)

def parse_test_dataset(directory):
    ids = []
    X = []

    with open(directory, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for row in csv_reader:
            ids.append(int(row[0]))
            X.append(np.array(row[:14], dtype=float))

    return np.array(ids), np.array(X)


def load_dataset(train_directory, test_directory):

    train_X, train_Y =  parse_train_dataset(train_directory)
    test_ids, test_X = parse_test_dataset(test_directory)

    return train_X, train_Y, test_X, test_ids
  
