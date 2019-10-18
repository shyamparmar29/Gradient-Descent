# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:33:57 2019

@author: Royal Legion 04
"""
import numpy as np

X =[0.5, 2.5]
Y = [0.2, 0.9]

def activation_function(w,b,x):
    return 1 / (1 + np.exp(-(w*x + b)))  #sigmoid function

def loss_function(w,b):
    error = 0
    for (x,y) in zip(X,Y):
        fx = activation_function(w,b,x)
        error += 0.5 * (fx - y)**2
    return error

def grad_w(w,b,x,y):
    fx = activation_function(w,b,x)
    return (fx-y) * fx * (1-fx) * x

def grad_b(w,b,x,y):
    fx = activation_function(w,b,x)
    return (fx-y) * fx * (1-fx)

def do_gradient_descent():
    w,b,eta,max_epochs = -2,-2,1,1000
    for i in range(max_epochs):
        dw,db = 0,0
        for (x,y) in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)
        w = w - (eta * dw)
        b = b - (eta * db)
        print(w,b)
        
do_gradient_descent()