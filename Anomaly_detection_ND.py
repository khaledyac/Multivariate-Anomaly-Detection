# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:16:12 2019

@author: Khaled YACOUBI
"""
#%reset -f
#%clear

########################Building Anomaly Detection Model from Scratch########################

""" ------------------------------------ Libraries -------------------------------------"""
# Import libraries
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import math
tic = time.time()

""" ------------------------------- Data Pre-processing --------------------------------"""
# Import dataset
dataset = sio.loadmat('anomalyData.mat')
X = dataset['X']
Xval = dataset['Xval']
yval = dataset['yval']

# Visualizing dataset
plt.scatter(X[:, 0], X[:, 1], marker = "x")
plt.xlabel('Latency(ms)')
plt.ylabel('Throughput(mb/s)')

""" ------------------------------ Gaussian Distribution -------------------------------"""
# Concept Explanation
"""To perform anomaly detection, you will first need to fit a model 
to the data’s distribution. Given a training set {x(1), …, x(m)} 
(where x(i) ∈ R^n, here n = 2), you want to estimate the Gaussian distribution 
for each of the features. For each feature (i = 1 . . . n), 
you need to find parameters mean and variance(mu, sigma²). 
For doing that let’s write down the function that calculates the mean and variance 
of the array(or you can call it matrix) X."""

# Estimation of the Gaussian Distribution for each features
def estimateGaussian(X):
    n = np.size(X, 1)
    m = np.size(X, 0)
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))
    
    mu = np.reshape((1/m)*np.sum(X, 0), (1, n))
    sigma2 = np.reshape((1/m)*np.sum(np.power((X - mu),2), 0),(1, n))
    
    return mu, sigma2

# Execution of the fuction above
mu, sigma2 = estimateGaussian(X)

""" ----------------------- Multivariate Gaussian Distribution ------------------------"""
# Concept Explanation
"""The multivariate Gaussian is used to find the probability of each example 
and based on some threshold value we decide whether to flag an anomaly or not."""

"""Here, mu is the mean of each feature and the variable sigma calculates 
the covariance matrix. These two parameters are used to calculate the probability p(x). 
‘e’ is the threshold value that we are going to discuss in detail further. 
Once you understand the expression, the code is very simple to implement. 
Let’s see how to put it into code."""

# Multivariate Gaussian Distribution
def multivariateGaussian(X, mu, sigma2):
     n = np.size(sigma2, 1)
     m = np.size(sigma2, 0)
     #print(m,n)
     
     if n == 1 or m == 1:
        # print('Yes!')
         sigma2 = np.diag(sigma2[0, :])
     #print(sigma2)
     X = X - mu
     pi = math.pi
     det = np.linalg.det(sigma2)
     inv = np.linalg.inv(sigma2)
     val = np.reshape((-0.5)*np.sum(np.multiply((X@inv),X), 1),(np.size(X, 0), 1))
     #print(val.shape)
     p = np.power(2*pi, -n/2)*np.power(det, -0.5)*np.exp(val)
     
     return p

# Execution of the fuction above 
p = multivariateGaussian(X, mu, sigma2)
print('\n\nsome values of P are:',p[1],p[23],p[45],p.shape)
pval = multivariateGaussian(Xval, mu, sigma2)
print('\n\nsome values of P are:',pval[1],pval[23],pval[45],pval.shape)

# Explanation of the method
"""First, we find the stepsize to have a wide range of threshold values 
to decide the best one. We use the F1 score method to determine the best parameters 
i.e bestepsilon and bestF1. Predict anomaly if pval<epsilon 
that gives a vector of binary values in the variable predict. 
F1 score takes into consideration precision and recall.

In line number 19 I’ve implemented a for loop to calculate the tp, fp, and fn.
I’d love to hear from you if you could come out with some vectorised 
implementation for the Logic."""

# Finding the best epsilon and calculating the confusion matrix
def selectThreshHold(yval, pval):
    
    F1 = 0
    bestF1 = 0
    bestEpsilon = 0
    
    stepsize = (np.max(pval) - np.min(pval))/1000
        
    epsVec = np.arange(np.min(pval), np.max(pval), stepsize)
    noe = len(epsVec)
    
    for eps in range(noe):
        epsilon = epsVec[eps]
        pred = (pval < epsilon)
        prec, rec, ac = 0,0,0
        tp,fp,fn,tn = 0,0,0,0
        
        try:
            for i in range(np.size(pval,0)):
                if pred[i] == 1 and yval[i] == 1:
                    tp+=1
                elif pred[i] == 1 and yval[i] == 0:
                    fp+=1
                elif pred[i] == 0 and yval[i] == 1:
                    fn+=1
                elif pred[i] == 0 and yval[i] == 0:
                    tn+=1
            prec = tp/(tp + fp)
            rec = tp/(tp + fn)
            F1 = 2*prec*rec/(prec + rec)
            ac = (tp+tn)/(tp+tn+fn+fp)
            if F1 > bestF1:
                bestF1 = F1
                bestEpsilon = epsilon
        except ZeroDivisionError:
            print('Warning dividing by zero!!')          
       
    return bestF1, bestEpsilon

F1, epsilon = selectThreshHold(yval, pval)
print('Epsilon and F1 are:',epsilon, F1)

# Generating the outliers after finding the best epsilon
outl = (p < epsilon)
def findIndices(binVec):
    l = []
    for i in range(len(binVec)):
        if binVec[i] == 1:
            l.append(i)
    return l
listOfOutliers = findIndices(outl)    

count_outliers = len(listOfOutliers)
print('\n\nNumber of outliers:', count_outliers)
print('\n',listOfOutliers)

# Visualizing the outliers
plt.scatter(X[:, 0], X[:, 1], marker = "x")
plt.xlabel('Latency(ms)')
plt.ylabel('Throughput(mb/s)')
plt.scatter(X[listOfOutliers,0], X[listOfOutliers, 1], facecolors = 'none', edgecolors = 'r')
plt.show()

# Executing the model on the test data set
""" ------------------------------------ Data set -------------------------------------"""
newDataset = sio.loadmat('anomalyDataTest.mat')
Xtest = newDataset['X']
Xvaltest = newDataset['Xval']
yvaltest = newDataset['yval']
print(Xtest.shape,Xvaltest.shape,yvaltest.shape)


# Applying PCA
from sklearn.decomposition import PCA
"""pca = PCA(n_components= None)
In order to get the percentage of the explained variables of the initial ones 
before choosing which components we are going to use in our problem modeling"""
pca = PCA(n_components= 11)
pca.fit(Xtest)
Xtest = pca.transform(Xtest)
Xvaltest = pca.transform(Xvaltest)

# We'll repeat the same steps as above but simply on a larger dataset
mutest, sigma2test = estimateGaussian(Xtest)
#print('\nmutest:\n',mutest)
#print('\nsigma2test:\n',sigma2test)
ptest = multivariateGaussian(Xtest, mutest, sigma2test)
#print('\nptest[300]\n',ptest[300])
pvaltest = multivariateGaussian(Xvaltest, mutest, sigma2test)
#print('\npval[45]\n',pval[45])

F1test, epsilontest = selectThreshHold(yvaltest, pvaltest)
print('\nBest epsilon and F1 are\n',epsilontest, F1test)

# Outliers
outliersTest = ptest < epsilontest
listOfOl = findIndices(outliersTest)

print('\n\n Outliers are:\n',listOfOl)
print('\n\nNumber of outliers are: ',len(listOfOl))
toc = time.time()
print('\n\nTotal time taken: ',str(toc - tic),'sec')

# Readme File
with open('Readme.txt','w+') as file:
    content = file.read()
    file.write()