import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import scipy as sc
"""
Import the DecisionTreeClassifier model.
"""
#Import the DecisionTreeClassifier

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
train_features = dataset.iloc[:160,:2]
test_features = dataset.iloc[160:,:2]
train_targets = dataset.iloc[:160,2]
test_targets = dataset.iloc[160:,2]
print(train_features)
###########################################################################################################
##########################################################################################################
"""
Train the model
"""
clf = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)

###########################################################################################################
##########################################################################################################
"""
Predict the classes of new, unseen data
"""
prediction = clf.predict(test_features)
###########################################################################################################
##########################################################################################################
"""
Check the accuracy
"""
print("The prediction accuracy is: ",clf.score(test_features,test_targets)*100,"%")
print(clf)
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
        out_file=dot_data,
        feature_names=None,
        class_names=None,
        filled=True, rounded=True,
        impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("coil02.pdf")