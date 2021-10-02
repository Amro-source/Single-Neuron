# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:53:58 2021

@author: Zikantika
"""


import numpy as np
import pandas as pd

def nn_batch(x,y,w,lr,n_epoch):
    cost_list = []
    y_hat = np.dot(x,w)  # calculate y_hat for first iteration
    for epoch in range(100):
        dcostdw1 = 2*0.5*np.dot(-x[:,0],(y-y_hat))  # matrix multiplication of dcostdw1 = [-x1,-x3] (1x2) * [[y1-y_hat1],[y2-y_hat2]] (2x1) 
        dcostdw2 = 2*0.5*np.dot(-x[:,1],(y-y_hat))  # matrix multiplication of dcostdw2 = [-x2,-x4] (1x2) * [[y1-y_hat1],[y2-y_hat2]] (2x1) 
        w = w - [[lr*dcostdw1[0]],
                 [lr*dcostdw2[0]]]    # w = w (2x1) - dcostdw (2x1)
        y_hat = np.dot(x,w)   # calculate new y_hat using new weights after above steps
        cost = 0.5*np.sum((y - y_hat)**2)  # calculate new cost 
        if epoch%20 ==0:
            print("epoch :{:d} w1 :{:f} w2 :{:f} cost:{:f}".format(epoch,w[0][0],w[1][0],cost))
        cost_list.append([epoch,cost])
        if cost <= 0.0001:
            break

    cost_list = pd.DataFrame(cost_list,columns=['epoch','cost'])
    return w,cost_list



def nn_batch_generalized(x,y,w,lr,n_epoch):
    cost_list = []  # initialize list to store epochs and cost
    y_hat = np.dot(x,w)  # calculate y_hat for first iteration
    for epoch in range(n_epoch):  
        dcostdw = 2*0.5*np.dot(-x.T,(y-y_hat))   # notice change from previous function "nn_batch". 
                                                #  Matrix multiplication dcostdw (3x1) = tranpose(X) (3x4)  * (y - y_hat) (4x1) 
                                                # above matrix shapes correspond to below example. for other example shapes may change but this function will work withput any changes needed.
        w = w - lr*dcostdw  # w new (3x1) = w (3x1) - dcostdw (3x1)
        y_hat = np.dot(x,w) # calculate new y_hat using new weights after above steps
        cost = 0.5*np.sum((y - y_hat)**2)  # calculate new cost
        cost_list.append([epoch,cost])
        if epoch%1000==0:
            print("epoch :{:d} cost:{:f}".format(epoch,cost))
        if cost <= 0.001:
            print("epoch :{:d} cost:{:f}".format(epoch,cost))
            break

    cost_list = pd.DataFrame(cost_list,columns=['epoch','cost'])
    return w,cost_list


def nn_batch(x,y,w,lr,n_epoch):
    cost_list = []
    y_hat = np.dot(x,w)  # calculate y_hat for first iteration
    for epoch in range(100):
        dcostdw1 = 2*0.5*np.dot(-x[:,0],(y-y_hat))  # matrix multiplication of dcostdw1 = [-x1,-x3] (1x2) * [[y1-y_hat1],[y2-y_hat2]] (2x1) 
        dcostdw2 = 2*0.5*np.dot(-x[:,1],(y-y_hat))  # matrix multiplication of dcostdw2 = [-x2,-x4] (1x2) * [[y1-y_hat1],[y2-y_hat2]] (2x1) 
        w = w - [[lr*dcostdw1[0]],
                 [lr*dcostdw2[0]]]    # w = w (2x1) - dcostdw (2x1)
        y_hat = np.dot(x,w)   # calculate new y_hat using new weights after above steps
        cost = 0.5*np.sum((y - y_hat)**2)  # calculate new cost 
        if epoch%20 ==0:
            print("epoch :{:d} w1 :{:f} w2 :{:f} cost:{:f}".format(epoch,w[0][0],w[1][0],cost))
        cost_list.append([epoch,cost])
        if cost <= 0.0001:
            break

    cost_list = pd.DataFrame(cost_list,columns=['epoch','cost'])
    return w,cost_list


x = np.array([[3,8],
             [10,7]])
y = np.array([[41],[58]])
lr = 0.001
w = np.ones((x.shape[1],1))
lr = 0.001
n_epoch=100

w,cost_list = nn_batch(x,y,w,lr,n_epoch)
#print ("y_hat_final:\n{:s}".format(np.dot(x,w)))
#---------------------------------------------------




x = [3,8]   
y = 41
w1_init = 0
w2_init = 1
lr = 0.01
n_epochs = 100
#w1,w2,cost_list = nn(x,y,w1_init,w2_init,lr,n_epochs)
#w = [[w1],[w2]]
#print(np.dot(x,w))




  
x = np.array([[3,8,10],
             [10,7,2],
             [2,9,15],
             [1,12,34],
             [3,1,9]])
y = [[51],[60],[57],[85],[22]]
w = np.ones((x.shape[1],1))
lr = 0.0001
n_epoch=10000
w,cost_list = nn_batch_generalized(x,y,w,lr,n_epoch)
#print "y_hat_final:\n{:s}".format(np.dot(x,w))
#print "trained weights :\n{:s}".format(w)

#----------------------------------------