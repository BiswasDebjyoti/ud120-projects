#!/usr/bin/python
from sklearn import svm
from sklearn.metrics import accuracy_score

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

svf = svm.SVC(C=10000,kernel="rbf")

svf.fit(features_train,labels_train)
pred= svf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)

print accuracy

#########################################################


