#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 23:04:54 2022

@author: Ahmet Yaztürk 
Based on: https://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/
"""
import numpy as np
import matplotlib.pyplot as plt

dataset = np.array([
    #X1		X2		Y
    [2.7810836, 2.550537003,	0],
    [1.465489372, 2.362125076,	0],
    [3.396561688	,4.400293529,	0],
    [1.38807019,	1.850220317,	0],
    [3.06407232,	3.005305973,	0],
    [7.627531214	,2.759262235	,1],
    [5.332441248	,2.088626775,	1],
    [6.922596716	,1.77106367,	1],
    [8.675418651, -0.2420686549,	1],
    [7.673756466	, 3.508563011,	1],
    [5.0,   2.5,    0]])

#%%

class LogReg:
    def __init__(self, dataset, n_epoch, w=None, b=0, a=0.5):
        if w == None:
            self.w = np.zeros(len(dataset[0])-1)
        else:
            self.w = w
        self.dataset, self.epoch, self.b, self.a = dataset, n_epoch, b, a
        for i in range(n_epoch):
            self._epoch()
    
    def _epoch(self):
        for row in self.dataset:
            z = np.dot(row[:-1],self.w) + self.b
            p = self._sigmoid(z)
            self.w += (self.a * (row[-1] - p) * p * (1-p)) * row[:-1]
            self.b += self.a * (row[-1] - p) * p * (1-p)
    def _sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def predict(self, x, binary=False):
        z = np.dot(self.w,x) + self.b
        p = self._sigmoid(z)
        if binary:
            if p>=0.5:
                return 1
            else:
                return 0
        else:
            return p


#%%
model = LogReg(dataset, n_epoch=15)

w,b = model.w, model.b

xpoints = np.arange(1,9,0.1)
def find_ypoints(x0):
    return (w[0]*x0+b)/(-w[1])

bluex, bluey, redx, redy = [], [], [], []
for row in dataset:
    if row[2] == 0:
        bluex.append(row[0])
        bluey.append(row[1])
    elif row[2] == 1:
        redx.append(row[0])
        redy.append(row[1])
    else:
        print("Third column contains nonbinary value?")
        break


plt.scatter(bluex, bluey,color="blue")
plt.scatter(redx, redy, color="red")
ypoints = find_ypoints(xpoints)
plt.plot(xpoints,ypoints,ls="--",color="gray")

plt.show()
plt.close()
print("Coefficients:",w[0],w[1],b)
