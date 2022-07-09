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
    [5.0,   2.5, 0]])

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
plt.show()
#%%
def sigmoid(x):
    return 1/(1+np.exp(-x))

w0,w1,b = 0,1,0
def prob(x0,x1):
    output = w0*x0 + w1*x1 + b
    return sigmoid(output)

alpha = 0.3 #learningrate
def learning(w,pred,exp,input):
    return w + alpha * (exp-pred) * pred * (1-pred) * input

def epoch():
    global w0,w1,b
    for row in dataset:
        pred = prob(row[0],row[1])
        w0 = learning(w0,pred,row[2],row[0])
        w1 = learning(w1,pred,row[2],row[1])
        b  = learning(b, pred,row[2],1)

def find_x1(x0):
    return (w0*x0+b)/(-w1)

xpoints = np.arange(1,9,0.1)
plt.figure(figsize=(18,12))
for j in range(15):
    # Uncomment if you want to color the spots by prediction
    # bluex, bluey, redx, redy = [], [], [], []
    # for row in dataset:
    #     pred = prob(row[0],row[1])
    #     if pred < 0.5 and pred >= 0:
    #         bluex.append(row[0])
    #         bluey.append(row[1])
    #     elif pred >= 0.5 and pred <= 1:
    #         redx.append(row[0])
    #         redy.append(row[1])
    #     else:
    #         print("Probability outside 0-1 range?")
    #         print(row)
    #         break
    plt.subplot(3,5,j+1)
    plt.scatter(bluex, bluey,color="blue")
    plt.scatter(redx, redy, color="red")
    ypoints = find_x1(xpoints)
    plt.plot(xpoints,ypoints,ls="--",color="gray")
    print(w0,w1,b)
    epoch()
plt.show()
plt.close()
print("Final coefficients:",w0,w1,b)
