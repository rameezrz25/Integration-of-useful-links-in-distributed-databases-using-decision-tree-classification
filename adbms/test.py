import numpy as np
import pandas as pd
import scipy as sc
import importlib
"""
Import the DecisionTreeClassifier model.
"""
#Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
###########################################################################################################
##########################################################################################################
"""
Import the Zoo Dataset
"""
#Import the dataset 
dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt',sep= '\t', header= None)
#dataset = pd.read_csv(https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt',sep= '\t', header= None)
#print(dataset)
#We drop the animal names since this is not a good feature to split the data on
#dataset=dataset.drop(0,axis=1)
###########################################################################################################
##########################################################################################################
"""
Split the data into a training and a testing set
"""
print(dataset.iloc[:,0])
train_features = dataset.iloc[:80,:-1]
test_features = dataset.iloc[80:,:-1]
train_targets = dataset.iloc[:80,-1]
test_targets = dataset.iloc[80:,-1]
print(train_features)
###########################################################################################################
##########################################################################################################
"""
Train the model
"""
tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)

###########################################################################################################
##########################################################################################################
"""
Predict the classes of new, unseen data
"""
prediction = tree.predict(test_features)
###########################################################################################################
##########################################################################################################
"""
Check the accuracy
"""
print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")
print(tree)