import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False


def classify_data(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]

    return classification


def calculate_entropy(data):
    label_column = data.iloc[:,-1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt',sep= '\t', header= None)
#df = df.drop("Id", axis=1)
#df = df.rename(columns={"class": "label"})
#df.head()
indices = df.index.tolist()
print(indices)
print(df)
random.seed(0)
train_df, test_df = train_test_split(df, test_size=2000)
data = train_df.values
ent=calculate_entropy(df)
print(ent)
