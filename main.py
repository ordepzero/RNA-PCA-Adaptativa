# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:35:10 2016

@author: PeDeNRiQue
"""

import math
import random
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def read_file(filename):
    array = []
    
    with open(filename,"r") as f:
        content = f.readlines()
        for line in content: # read rest of lines
            array.append([x for x in line.split(",")])   
    return np.array(array);
    
def normalize_data(f,has_target=True):
    
    x = np.array(f)
    x_normed = (x - x.min(axis=0))/ (x.max(axis=0) - x.min(axis=0))
    
    #SUBSTITUIO OS VALORES ALVO DA ULTIMA COLUNA NOS DADOS NORMALIZADOS
    if(has_target):
        x_normed[:,-1] = f[:,-1]

    return x_normed    
    
def change_class_name(data,dic):
    for x in range(len(data)):
        data[x][-1] = dic[data[x][-1]]
    return data

def str_to_number(data):
    return[[float(j) for j in i] for i in data]

def convert(values):
    position = 0;
       
    for i in range(len(values)):
        if(values[i] > values[position]):
            position = i
    result = [0]*3
    result[position] = 1
    return result
    
class Neuron:
    
    def __init__(self,n_weights):
        self.weights = [random.uniform(0, 1) for n in range(n_weights) ]
        #print(self.weights)
    
    def calculate_output(self,entries):
        #print(self.weights,entries)
        self.entries = entries
        self.output = np.dot(entries, np.transpose(self.weights))
        
        return self.output
    
    def update_weight(self,new_weigths):
        self.weights = new_weigths

class Net:
    
    
    def __init__(self):
        self.lr = 0.001
        pass
    
    def execute(self,data):
        n_neurons = 2
        n_entries = 4
        self.neurons = [Neuron(n_entries) for n in range(n_neurons)] 
        
        
        for epoch in range(500):
            
            d = random.randint(0, 149)  
            entry = data["input"][d]
            self.outputs = [n.calculate_output(entry) for n in self.neurons]
            
            #self.outputs = [[n.calculate_output(d) for n in self.neurons] for d in data["input"]]
            
            #print(self.outputs)
            
            self.update_weights(entry)
        
        return [n.weights for n in self.neurons]
        
        
        
        
    def update_weights(self,entry):
        #[print(n.weights_output) for n in self.neurons] 
        new_weights = []
        for n in range(len(self.neurons)):
            weights = self.neurons[n].weights
            output = self.neurons[n].output
            part = 0
                
            for w in range(len(weights)):
                for i in range(n):
                    part += self.neurons[i].weights[w] * self.neurons[i].output
                
                weights[w] = weights[w] + (self.lr * ((output * entry[w]) - (output*part)))
                
            #print(weights)
            #self.neurons[n].update_weight(weights)
            new_weights.append(weights)    
            
        for nw in range(len(new_weights)):
            norm = [float(i)/max(new_weights[nw]) for i in new_weights[nw]]
            print(norm)
            
            self.neurons[nw].update_weight(norm)
  

def show_tables(data,targets):
    
    data = np.transpose(data)    
    
    for ind in range(len(targets)):    
        if(targets[ind] == 0): 
            plt.plot(data[0][ind], data[1][ind], 'ro')
        elif(targets[ind] == 1):
            plt.plot(data[0][ind], data[1][ind], 'bo')
        else:
            plt.plot(data[0][ind], data[1][ind], 'go')
    plt.show()
            
if __name__ == "__main__":
    
    dic = {'Iris-setosa\n': 0, 'Iris-versicolor\n': 1, 'Iris-virginica\n': 2}    
    
    filename = "iris.txt"
    file = read_file(filename)
    file = change_class_name(file,dic)
    file = str_to_number(file)
    file_array = np.array(file)
    #data = normalize_data(file_array)
    data = {"input": file_array[:,:-1], "target":file_array[:,-1]}
    data_test = {"input": data["input"][90:150], "target":data["target"][90:150]} #data[90:150]
    data_train = {"input": data["input"][:90], "target":data["target"][:90]}  
    
    #print(data)
    
    net = Net()
    
    eigen = net.execute(data)
    
    print(eigen)
    
    new_data = []
    
    for d in data["input"]:
        new_d = [np.dot(d,eigen[0]),np.dot(d,eigen[1])]
        #print(np.dot(d,eigen[0]),np.dot(d,eigen[1]))
        new_data.append(new_d)
    show_tables(new_data,data["target"])
    