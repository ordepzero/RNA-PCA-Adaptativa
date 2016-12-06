# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:47:14 2016

@author: PeDeNRiQue
"""

a = [0.02920384424670386, 0.5637064928298525, 0.4536177480363561, 0.25780503759269535] 
b = [ 6.8,  3.2,  5.9,  2.3]

soma = 0
for i in range(4):
    soma += a[i]*b[i]
    
print(soma)