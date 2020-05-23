import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from SDNE import SDNE
from optparse import OptionParser
import os
import argparse
from Values import Values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import networkx as nx
import matplotlib.pyplot as plt 
from sklearn.multiclass import OneVsRestClassifier
from Classifying_Model import Classifying_Model 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='data')
parser.add_argument('--save',type=str,default='/')
parser.add_argument('--hidden_dim',type=int,default=16,help='Hidden dimension')
parser.add_argument('--input_dropout',type=float,default=0.5,help='Input Dropout rate')
parser.add_argument('--dropout',type=float,default=0.5,help='Dropoutrate' )
parser.add_argument('--batch_size',type=int,default= 32,help='batch_size')
parser.add_argument('--optimizer',type=str,default='adam',help='Optimizer')
parser.add_argument('--learning_rate',type=float,default=0.01,help='Learning Rate')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--self_link_weight',type=float,default=1.0,help='Weight')
parser.add_argument('--pre_epoch', type=int, default=100, help='Number of pre-training epochs.')
parser.add_argument('--epoch',type=int,default=100,help='Epoch')
parser.add_argument('--iter',type=int,default=10,help='iterations')
parser.add_argument('--use_gold',type=int,default=1,help='Whether using the gold')
parser.add_argument('--tau',type=float,default=1.0,help='Temperature')
parser.add_argument('--draw',type=str,default='max',help='Drawing sampling')
parser.add_argument('--depth',type=float,default=1.0,help='Predicting neighbors within [depth] steps.')
parser.add_argument('--neg_sample',type=int,default=0.0,help='ng_sample_ratio') 
parser.add_argument('--seed',type=int,default=1,help='seed')
parser.add_argument('--reg',type=int,default=1, help='reg_value')
parser.add_argument('--alpha',type=int,default=100, help='alpha_value')
parser.add_argument('--dbn_init',type=bool,default=True, help='dbn_init')
parser.add_argument('--dbn_epochs',type=int,default=50, help='dbn_epochs')
parser.add_argument('--dbn_batch_size',type=int,default=32, help='dbn_batch_size')
parser.add_argument('--dbn_learning_rate',type=int,default=0.1, help='dbn_learning_rate')
parser.add_argument('--gamma',type=int,default=1, help='gamma_value')
parser.add_argument('--beta',type=int,default=50, help='beta value')
parser.add_argument('--display',type=int,default=5, help='display value') 
parser.add_argument('--struct',type=list,default=[-1,100], help='structvalue') 
parser.add_argument('--cpumizer', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()





files = vars(args)









train_graph_file = files['dataset'] + 'E:/KDD/public-data/public/e/train.data/edge.tsv'
train_label_file = files['dataset'] + 'E:/KDD/public-data/public/e/train.data/train_label.txt'
train_node_id = files['dataset'] + 'E:/KDD/public-data/public/e/train.data/train_node_id.txt'
features  = files['dataset'] + 'E:/KDD/public-data/public/e/train.data/feature.tsv'
test_label = files['dataset'] + 'E:/KDD/public-data/public/e/test_label.txt'
test_node_id = files['dataset'] + 'E:/KDD/public-data/public/e/train.data/test_node_id.txt'


files['struct'][0] = 7521
train_graph_data = Values(train_graph_file,files['neg_sample'])
print("AAAAA")
print(train_graph_data)



model = SDNE(files,train_graph_file)
model.do_variables_init(train_graph_data)
embedding  = None
feature_embedding = None
daata = pd.read_csv('E:/KDD/public-data/public/e/train.data/edge.tsv',delimiter='\t')
graph = nx.from_pandas_edgelist(daata,'src_idx','dst_idx',['edge_weight'])
adj = nx.adjacency_matrix(graph).todense()

while(True):
  mini_batch  = train_graph_data.sample(files['batch_size'],do_shuffle=False)
  if embedding is None:
    embedding = model.get_embedding(mini_batch)
  else:
    embedding = np.vstack((embedding,model.get_embedding(mini_batch)))
  if train_graph_data.is_epoch_end:
    break
dict = {}
def all_indices(value,list):
  ind = []
  idx = -1
  while True:
    try:
      idx = list.index(value,idx+1)
      ind.append(idx)
    except ValueError:
      break
  return ind
Da = list(daata['src_idx'])
Daa = list(daata['dst_idx'])
for i in Da:
  ind = all_indices(i,Da)
  Li = []
  for j in ind:
    Li.append(Daa[j])
  dict[i] = Li
  
print(dict)
epochs =0
def ddata(Emb,Dict):
  Message_Passage_Embedding = Emb
  for i in range(len(Emb)):
    sum=list(np.zeros(100))
    La = dict[i]
    for j in range(len(Emb[i])):
      for k in La:
        sum[j] = sum[j] + embedding[k][j]
    for l in range(len(Emb[i])):
      Message_Passage_Embedding[i][l] = Emb[i][l]+sum[l]
  Message = Message_Passage_Embedding
  for i in range(len(Emb)):
    La = dict[i]
    for j in La:
      for k in range(len(Emb[i])):
        Message[j][k] = Message_Passage_Embedding[j][k]+Message_Passage_Embedding[i][k]
  return Message,Message_Passage_Embedding
Train_label = pd.read_csv('E:/KDD/public-data/public/e/train.data/train_label.tsv',delimiter='\t')
Test_label = pd.read_csv('E:/KDD/public-data/public/e/test_label.tsv',delimiter='\t')
Train_labelled_id = np.array(Train_label['node_index'])
Test_labelled_id = np.array(Test_label['node_index'])
train_label = np.array(Train_label['label'])
test_label = np.array(Test_label['label'])
while(True):
  if train_graph_data.is_epoch_end:
    loss = 0
    Half_Message_Passage_Labelled_embedding = []
    Half_Message_Passage_UnLabelled_embedding = []
    Full_Message_Passage_Labelled_embedding = []
    Full_Message_Passage_UnLabelled_embedding = [] 
    if epochs % files['display']==0:
      embedding = None
      while(True):
        mini_batch = train_graph_data.sample(files['batch_size'],do_shuffle = False)
        loss+=model.get_loss(mini_batch)
        if embedding is None:
          embedding = model.get_embedding(mini_batch) 
        else:
          embedding = np.vstack((embedding,model.get_embedding(mini_batch)))
        if train_graph_data.is_epoch_end:
          break
      Half_Message_Passage_Embedding,Full_Message_Passage_Embedding =  ddata(embedding,dict)
      for i in Train_labelled_id:
        Half_Message_Passage_Labelled_embedding.append(Half_Message_Passage_Embedding[i])
      for i in Test_labelled_id:
        Half_Message_Passage_UnLabelled_embedding.append(Half_Message_Passage_Embedding[i])
      for i in Test_labelled_id:
        Full_Message_Passage_UnLabelled_embedding.append(Full_Message_Passage_Embedding[i])
      for i in Train_labelled_id:
        Full_Message_Passage_Labelled_embedding.append(Full_Message_Passage_Embedding[i])
      Classifying_Model(Half_Message_Passage_Labelled_embedding,train_label,Half_Message_Passage_UnLabelled_embedding,test_label)
      Classifying_Model(Full_Message_Passage_Labelled_embedding,train_label,Full_Message_Passage_UnLabelled_embedding,test_label) 
    if epochs == files['epoch']:
      print("exceed epochs limit terminating")
      break
    epochs+=1
  mini_batch = train_graph_data.sample(files['batch_size'])
  loss = model.fit(mini_batch) 
     
       
      
        


