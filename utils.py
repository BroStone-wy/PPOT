#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:48:53 2022

@author: lnt2
"""
import torch
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
from ot.utils import unif, dist, list_to_array
from ot.backend import get_backend
import warnings
from scipy.optimize.linesearch import scalar_search_armijo
from ot.utils import dist as dist
import os.path as osp
import os


numItermax = 1000
stopThr = 1e-9

def computeTransportSinkhorn(distributS, distributT, M, reg, numItermax=1000, stopThr=1e-9):
    dim_S = len(distributS)
    dim_T = len(distributT)
    
    K = np.exp(M / -reg)
    Kp = np.dot(np.diag(1/distributS), K)
    
    #  u initialization
    u = np.ones(len(distributS)) / dim_S
    v = np.ones(len(distributT)) / dim_T
    
    err = 1
    for i in range(numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = distributT / KtransposeU
        u = 1. / np.dot(Kp, v)
        if (np.any(KtransposeU == 0)
                    or np.any(np.isnan(u)) or np.any(np.isnan(v))
                    or np.any(np.isinf(u)) or np.any(np.isinf(v))):
                # we have reached the machine precision
                # come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration %d' % i)
            u = uprev
            v = vprev
            break
        if i % 10 == 0:
                transp = np.dot(np.diag(u), np.dot(K, np.diag(v)))#高维度张量求和
                err = np.linalg.norm(np.sum(transp, axis=0) - distributT)**2
                if err < stopThr:
                    break
    # P = u.reshape((-1, 1)) * K * v.reshape((1, -1))
    return np.dot(np.diag(u), np.dot(K, np.diag(v)))





def  gcg(distributS, distributT, M, reg1, reg2, f, df,G0=None,
        numItermax=100,numInnermax=100, stopThr=1e-9, stopThr2=1e-9):
    
    if G0 is None:
        G = np.outer(distributS, distributT)
    else:
        G = G0
    def cost(G):
        return np.sum(M * G) + reg1 * np.sum(G * np.log(G)) + reg2 * f(G)      #熵正则加上f的正则项目W距离。
    
    f_val = cost(G)
    loop = 1
    it = 0
    while loop:
 
        it += 1
        old_fval = f_val

        Mi = M + reg2 * df(G)
        if np.any(Mi) < 0:
            Mi += -np.min(Mi) + 1e-6
        # if it ==1:
        #     print(reg2 * df(G))
        #     print(M)
        # asa==1
        # 

        Gc = computeTransportSinkhorn(distributS, distributT, Mi, reg1, numItermax=numInnermax,stopThr=1e-9)
        deltaG = Gc - G
       
        dcost = Mi + reg1 * (1 + np.log(G)) 
        alpha, fc, f_val = line_search_armijo(cost, G, deltaG, dcost, old_fval,alpha_min=0., alpha_max=1.)
        
        G = G + alpha * deltaG
        
        if it >= numItermax:
            loop = 0
        
        abs_delta_fval = abs(f_val - old_fval)
        ralative_delta_fval = abs_delta_fval / abs(f_val)
        
        if ralative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0
    # print(it)
            
    return G

def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=1e-4,
                       alpha0=0.99, alpha_min=0, alpha_max=1):
    fc = [0]
    
    def phi(alpha1):
        fc[0] += 1
        
        return f(xk + alpha1 * pk, *args)
    
    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval
        
    derphi0 = np.sum(pk * gfk)

    alpha, phi1 = scalar_search_armijo(
        phi, phi0, derphi0, c1=c1, alpha0=alpha0)#armijo 准则
 
    if alpha is None:
        return 0., fc[0], phi0
    else:
        if alpha_min is not None or alpha_max is not None:
           alpha = np.clip(alpha, alpha_min, alpha_max)
        return float(alpha), fc[0], phi1

def sinkhorn_R1reg_lab(a, b, M, reg, eta, numItermax=100,
                     numInnerItermax=100, stopInnerThr=1e-9, 
                     intra_class=None, inter_class=None,aux=None,aux1=None):
    
    Intra = np.ones((len(a),len(b))) * intra_class * aux
    Inter = np.ones((len(a),len(b))) * inter_class * aux1
    zero = np.zeros_like(M)

    def f(G):
        res = 0
        phi = (G - Inter) * (Intra - G)
        phi = np.where(phi>0, phi, zero)
        res += phi.sum()
        return res
    
    def df(G):
        d_phi = np.zeros(G.shape)
        phi = (G - Inter) * (Intra - G)
        d_phi = Inter + Intra - 2 * G
        W = np.zeros(G.shape)
        W = np.where(phi<0, zero, d_phi)
        return W
  
    return gcg(a, b, M, reg, eta, f, df, G0=None, numItermax=numItermax,
               numInnermax=numInnerItermax, stopThr=stopInnerThr)

def sinkhorn_R1reg(a, b, M, reg, eta=0.1, numItermax=10,
                     numInnerItermax=10, stopInnerThr=1e-9, 
                     intra_class=None, inter_class=None):
    
    Intra = np.ones((len(a),len(b))) * intra_class 
    Inter = np.ones((len(a),len(b))) * inter_class 
    zero = np.zeros_like(M)

    def f(G):
        res = 0
        phi = (G - Inter) * (Intra - G)
        phi = np.where(phi>0, phi, zero)
        
        # phi = np.where(phi>0,phi,zero)
        res += phi.sum()
        return res
    
    def df(G):
        d_phi = np.zeros(G.shape)
        phi = (G - Inter) * (Intra - G)
        d_phi = Inter + Intra - 2 * G
        d_phi = np.where(phi < 0 ,zero, d_phi)
        
        return d_phi

    return gcg(a, b, M, reg, eta, f, df, G0=None, numItermax=numItermax,
               numInnermax=numInnerItermax, stopThr=stopInnerThr)


# # A to W
# [27, 24, 21, 24, 10, 28, 27, 29, 29, 24, 29, 30, 30, 29, 30, 79, 80, 75, 76, 76, 74, 80, 78, 78, 72, 60, 80, 79, 79, 76, 51]
# [23, 16, 22, 9, 12, 24, 32, 14, 16, 15,
#  21, 21, 24, 15, 24, 12, 9, 8, 8, 9, 
#  4, 6, 9, 8, 12, 3, 7, 9, 7, 6, 6]

# # A to D
# [27, 24, 21, 24, 10, 28, 27, 29, 29, 24, 29, 30, 30, 29, 30, 79, 80, 75, 76, 76, 74, 80, 78, 78, 72, 60, 80, 79, 79, 76, 51]
# [9, 16, 19, 9, 12, 9, 10, 11, 12, 12, 10, 8, 19, 12, 24, 6, 3, 2, 3, 3, 3, 4, 6, 5, 3, 2, 5, 7, 6, 6, 4]

# # W to A
# [8, 6, 8, 3, 4, 9, 12, 5, 6, 5, 8, 8, 9, 5, 9, 34, 24, 21, 22, 25, 12, 16, 24, 21, 32, 8, 20, 24, 19, 18, 16]
# [73, 65, 57, 65, 28, 75, 72, 77, 77, 64, 79, 80, 80, 78, 80, 29, 30, 28, 28, 28, 27, 30, 29, 29, 27, 22, 30, 29, 29, 28, 19]

# # D to A
# [3, 6, 7, 3, 4, 3, 3, 4, 4, 4, 3, 3, 7, 4, 9, 17, 9, 6, 8, 8, 10, 12, 18, 14, 8, 5, 14, 20, 16, 17, 12]
# [73, 65, 57, 65, 28, 75, 72, 77, 77, 64, 79, 80, 80, 78, 80, 29, 30, 28, 28, 28, 27, 30, 29, 29, 27, 22, 30, 29, 29, 28, 19]
    
    
    



