import os
import sys
import yaml
import pandas as pd

params = yaml.safe_load(open('params.yaml'))['split']

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython split.py data-file\n")
    sys.exit(1)

# Test data set split ratio
split_ratio = params['ratio']

input = sys.argv[1]
output_train = os.path.join('data', 'prepared', 'train.csv')
output_valid = os.path.join('data', 'prepared', 'valid.csv')

def split(df, split_ratio):
    total_data = len(df)
    train_index = int(total_data*split_ratio)

    return df[:train_index], df[train_index:]

df = pd.read_csv(input)
train_df, valid_df = split(df, split_ratio)

os.makedirs(os.path.join('data', 'prepared'), exist_ok=True)
train_df.to_csv(output_train, index=False)
train_df.to_csv(output_valid, index=False)