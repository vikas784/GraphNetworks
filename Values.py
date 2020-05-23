import scipy.io as sio
import numpy as np
from utils import *
import random
import copy
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import random
from utils import Dotdict
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt





class Values(object):
  def __init__(self,file_path,ng_sample_ratio):
    self.suffix = file_path.split('.')[-1]
    self.st = 0
    self.is_epoch_end=False
    self.N = self.Nodes(file_path)
    print(self.N)
    self.E = self.edges(file_path)
    self.file_path = file_path
    print("lmn")
    self.add = file_path[4:]
    self.data = pd.read_csv(self.add,delimiter='\t')
    self.data = pd.read_csv(self.add,delimiter='\t')
    self.g =  nx.from_pandas_edgelist(self.data,'src_idx','dst_idx')
    self.adj_matrix = nx.adjacency_matrix(self.g).todense()
    self.adj = self.adj_matrix
    print("kk")
    if(ng_sample_ratio>0):
       print("lll")
    self.__negativeSample(int(ng_sample_ratio*self.E))
    print("aop")
    self.order = np.arange(self.N)
    print("Vertexes : %d  Edges : %d ngSampleRatio: %f" % (self.N, self.E, ng_sample_ratio))
    self.is_epoch_end= False
  def __negativeSample(self,neg):
     i=0
     while(i<neg):
       a = random.randint(0,self.N-1)
       b = random.randint(0,self.N-1)
       if(a==b or self.adj_matrix[a,b]==1):
         continue
       self.adj_matrix[a,b] = -1
       self.adj_matrix[b,a] = -1 
       i+=1   
  
  def Nodes(self,file_path):
    self.file_path = file_path
    self.add = file_path[4:]
    self.data = pd.read_csv(self.add,delimiter='\t')
    self.nodes = len(np.unique(self.data['src_idx']))
    return self.nodes
  
  def edges(self,file_path):
    self.file_path = file_path
    self.add = file_path[4:]
    self.data = pd.read_csv(self.add,delimiter='\t')
    self.edges = len(self.data)-1
    return self.edges
    
  def sample(self, batch_size, do_shuffle = True, with_label = False):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self.order[0:self.N])
            else:
                self.order = np.sort(self.order)
            self.st = 0
            self.is_epoch_end = False 
        mini_batch = Dotdict()
        en = min(self.N, self.st + batch_size)
        index = self.order[self.st:en]
        mini_batch.X = np.array(self.adj_matrix[index])
        mini_batch.adjacency_matriX = (np.array(self.adj_matrix)[index])[:][:,index]
        if with_label:
            mini_batch.label = self.label[index]
        if (en == self.N):
            en = 0
            self.is_epoch_end = True
        self.st = en
        return mini_batch    
    