# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:59:09 2019

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

def do_nesterov_accelerated_gradient_descent():
    
    w,b,eta,max_epochs = -2, -2, 1, 200
    prev_v_w, prev_v_b, gamma = 0,0,0.9
    
    for i in range(max_epochs):
        dw,db = 0,0
        
        #do partial updates
        v_w = gamma * prev_v_w
        v_b = gamma * prev_v_b
        
        for (x,y) in zip(X,Y):
            #calculate gradients after partial update
            dw += grad_w(w - v_w, b - v_b, x, y)
            db += grad_b(w - v_w, b - v_b, x, y)
            
        #now do the full update
        v_w = gamma * prev_v_w + eta * dw
        v_b = gamma * prev_v_b + eta * db
        w = w - v_w
        b = b - v_b
        prev_v_w = v_w
        prev_v_b = v_b
        print(w,b) 
        
do_nesterov_accelerated_gradient_descent() 