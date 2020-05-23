import pandas as pd
import tensorflow as tf
import numpy as np
import random
import time
import copy
from res_bol_mac import res_bol_mac



class SDNE:
  def __init__(self,args,path):
    self.suffix = path.split('.')[-1]
    self.is_variables_init = False
    self.args = args
    self.path = path
    config = tf.ConfigProto()
    self.sess = tf.Session(config=config)
     
    
    self.layers = len(args['struct'])
    self.struct = args['struct']
    self.sparse_dot = False
    self.W = {}
    self.b = {}
    struct = self.struct
    for i in range(self.layers - 1):
            name = "encoder" + str(i)
            self.W[name] = tf.Variable(tf.random.normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
    struct.reverse()
    for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.W[name] = tf.Variable(tf.random.normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
    self.struct.reverse()

 
    self.Adjacency_matrix = tf.placeholder("float", [None,None])
    self.struct = struct
    tf.compat.v1.global_variables_initializer()
    self.X = tf.placeholder("float",[None,struct[0]])
    self.X2 = tf.placeholder("float",[None,struct[0]])
    tf.compat.v1.global_variables_initializer()

    self.Encode_Decode_process()
    self.loss = self.__make_loss(args)
    self.optimizer = tf.compat.v1.train.RMSPropOptimizer(args['learning_rate']).minimize(self.loss)
    self.optimizer1 = tf.compat.v1.train.AdamOptimizer(args['learning_rate']).minimize(self.loss)

  def Encode_Decode_process(self):
    def Encoder(X):
      for i in range(self.layers-1):
        name = "encoder" + str(i)
        X = tf.nn.sigmoid(tf.matmul(X,self.W[name])+self.b[name])
      return X
    def Decoder(X):
      for i in range(self.layers-1):
        name = "decoder"+ str(i)
        X = tf.nn.sigmoid(tf.matmul(X,self.W[name])+self.b[name])
      return X


  
    

    self.H = Encoder(self.X)
    self.X_reconstruct = Decoder(self.H)


  def __make_loss(self,args):
    def get1loss(Enc,adj_matrix):
      D = tf.linalg.tensor_diag(tf.reduce_sum(adj_matrix,1))
      L = D - adj_matrix
      return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(Enc),L),Enc))  
    def get2loss(X,Dec,beta):
      B = X * (beta-1) + 1
      return tf.reduce_sum(tf.pow((Dec - X)* B, 2))
    def get_reg_loss(w,b):
      ret = tf.add_n([tf.nn.l2_loss(wi) for wi in w.values()])
      ret = ret + tf.add_n([tf.nn.l2_loss(bi) for bi in b.values()])
      return ret


    self.loss_1 = get1loss(self.H,self.Adjacency_matrix)
    self.loss_2 = get2loss(self.X,self.X_reconstruct,self.args['beta'])
    self.loss_reg = get_reg_loss(self.W,self.b)
    return args['gamma']*self.loss_1 + args['alpha']*self.loss_2 + args['reg']*self.loss_reg

    
  def do_variables_init(self,data):
    def assign(a,b):
      op = a.assign(b)
      self.sess.run(op)
    init =  tf.compat.v1.global_variables_initializer()
    print("ABBBB")
    print(data)
    self.sess.run(init)
    if self.args['dbn_init']:
      shape  = self.struct
      myRBMs = []
      for i in range(len(shape)-1):
        myRBM = res_bol_mac([shape[i],shape[i+1]],{"batch_size":self.args['dbn_batch_size'],"learning_rate":self.args['dbn_learning_rate']},self)
        myRBMs.append(myRBM)
        for epoch in range(self.args['dbn_epochs']):
          error = 0
          for batch in range(0,self.struct[0],self.args['dbn_batch_size']):
            mini_batch = data.sample(self.args['dbn_batch_size']).X
            for k in range(len(myRBMs)-1):
              mini_batch = myRBMs[k].getH(mini_batch)
            error += myRBM.fit(mini_batch)
          print("rbm epochs:",epoch,"error:",error)
        W,bv,bh = myRBM.getWb()
        name = "encoder" + str(i)
        assign(self.W[name],W)
        assign(self.b[name],bh)
        name ="decoder" + str(self.layers - i -2)
        assign(self.W[name],W.transpose())
        assign(self.b[name],bv)
      self.is_Init = True
 
  def save_model(self,path):
        saver = tf.train.Saver(list(self.b.values()) + list(self.W.values()))
        saver.save(self.sess,path)

  def get_feed_dict(self,data):
    X = data.X
    return {self.X:data.X,self.Adjacency_matrix:data.adjacency_matriX}
  def __get_feed_dict(self,Encoded,adj):
    return {self.H:Encoded,self.Adjacency_matrix:adj}
  
  def get_loss(self, data):
        feed_dict = self.get_feed_dict(data)
        return self.sess.run(self.loss, feed_dict = feed_dict)
  

  def get_embedding(self,data):
    print(self.H,self.X)
    return self.sess.run(self.H,feed_dict= self.get_feed_dict(data))
  def fit(self,data):
    feed_dict = self.get_feed_dict(data)
    ret,_ = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
    return ret
  