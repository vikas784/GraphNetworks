import tensorflow as tf
import numpy as np


class res_bol_mac:
  def __init__(self,shape,para,path):
    self.para = para
    self.sess = tf.Session()
    stddev = 1.0/np.sqrt(shape[0])
    self.W = tf.Variable(tf.random_normal([shape[0],shape[1]],stddev=stddev),name = "Wii")
    self.bv = tf.Variable(tf.zeros(shape[0],name="a"))
    self.bh = tf.Variable(tf.zeros(shape[1]),name="b")
    self.v = tf.placeholder("float",[None,shape[0]])  
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)
    self.buildModel()
    pass


  def buildModel(self):
    self.h = self.sample(tf.sigmoid(tf.matmul(self.v,self.W)+self.bh))
    v_sample = self.sample(tf.sigmoid(tf.matmul(self.h,tf.transpose(self.W))+self.bv))
    h_sample = self.sample(tf.sigmoid(tf.matmul(v_sample,self.W)+self.bh))
    lr = self.para["learning_rate"] / tf.to_float(self.para["batch_size"])
    W_adder =  self.W.assign_add(lr*(tf.matmul(tf.transpose(self.v),self.h)-tf.matmul(tf.transpose(v_sample),h_sample)))
    bv_adder = self.bv.assign_add(lr*tf.reduce_mean(self.v-v_sample,0))
    bh_adder = self.bh.assign_add(lr*tf.reduce_mean(self.h-h_sample,0))
    self.upt = [W_adder,bv_adder,bh_adder]
    self.error = tf.reduce_sum(tf.pow(self.v-v_sample,2))
     

  def sample(self,probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs),0,1))
  

  def getH(self,data):
    return self.sess.run(self.h,feed_dict={self.v:data})
  

  def fit(self, data):
        _, ret = self.sess.run((self.upt, self.error), feed_dict = {self.v : data})
        return ret
  def getWb(self):
    return self.sess.run([self.W, self.bv, self.bh])