# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:47:14 2016

@author: PeDeNRiQue
"""
import numpy as np


z1 = [-1,1,-1,1,1,1,-1,1,-1]
z2 = [1,-1,1,-1,1,-1,1,-1,1]

zm1 = []
zm2 = []
results = []

for i in range(9):
    temp = []
    for j in range(9):
        temp.append(z1[i] * z1[j])
        print(z1[i] * z1[j],"  ",end="")
    zm1.append(temp)
    print()
    
print("--------------------------------------")
for i in range(9):
    temp = []
    for j in range(9):
        temp.append(z2[i] * z2[j])
        print(z2[i] * z2[j],"  ",end="")
    zm2.append(temp)
    print()
 
for i in range(9):
    temp = []
    for j in range(9):
        temp.append((zm1[i][j] + zm2[i][j])/9)
        print("{0:.2f}".format((zm1[i][j] + zm2[i][j])/9),"  ",end="")
    results.append(temp)  
    print()    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 