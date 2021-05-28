import os
import sys
import pandas as pd
import numpy as np

def separate_label_column(df):
    y = df["label"]
    x = train.drop(labels=["label"], axis=1)
    return x, y

def normalize(df):
    return df / 255.0

def convert_x_to_numpy(df_X, df_y):
    return df_X.values.reshape(-1,28,28,1), df_y.values

input = sys.argv[1]

input_train = os.path.join(input, 'train.csv')
input_valid = os.path.join(input, 'valid.csv')

output_train_x = os.path.join('data', 'preprocessed', 'train_X.npy')
output_train_y = os.path.join('data', 'preprocessed', 'train_y.npy')

output_valid_x = os.path.join('data', 'preprocessed', 'valid_X.npy')
output_valid_y = os.path.join('data', 'preprocessed', 'valid_y.npy')

os.makedirs(os.path.join('data', 'preprocessed'), exist_ok=True)

train = pd.read_csv(input_train)
train_X, train_y = separate_label_column(train)
train_X = normalize(train_X)
train_X, train_y = convert_x_to_numpy(train_X, train_y)
np.save(output_train_x, train_X)
np.save(output_train_y, train_y)

valid = pd.read_csv(input_valid)
valid_X, valid_y = separate_label_column(valid)
valid_X = normalize(valid_X)
valid_X, valid_y = convert_x_to_numpy(valid_X, valid_y)
np.save(output_valid_x, valid_X)
np.save(output_valid_y, valid_y)