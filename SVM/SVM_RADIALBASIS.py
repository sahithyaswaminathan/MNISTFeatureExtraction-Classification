# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 20:02:13 2018

@author: e0269724
"""

import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from svmutil import *
from svm import *

#Reading the train and test images:
images_train=mnist.train_images()
Xtrain = images_train.reshape((images_train.shape[0], images_train.shape[1] * images_train.shape[2]))

images_test=mnist.test_images()
Xtest= images_test.reshape((images_test.shape[0], images_test.shape[1] * images_test.shape[2]))

#Reading the train and test labels:
Ytrain=mnist.train_labels()
Ytest=mnist.test_labels()


scaler = StandardScaler()

#Scalar fit and transform the data:
Xtrain=scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

#PCA DIMENSION REDUCTION
pca=PCA(n_components=40) #[40,80,200]
Xtrain=pca.fit_transform(Xtrain)
Xtest = pca.transform(Xtest)

# Convert it into the list:
Xtrain=Xtrain.tolist()
Xtest=Xtest.tolist()
Ytrain=Ytrain.tolist()
Ytest=Ytest.tolist()

# SVM train and accuracy from the LIBSVM package:
model = svm_train(Ytrain, Xtrain, '-s 0 -t 2 -c 0.1')# RBFN Kernel
p_labs, p_acc, p_vals = svm_predict(Ytest, Xtest,model)