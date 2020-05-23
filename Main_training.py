import sklearn
from math import floor
from sklearn.metrics import accuracy_score
import numpy as np
import scipy




class Training:
    def __init__(self,name,base_classifier):
      self.name = name
      self.i = int(0)
      self.base_classifier = base_classifier
   

    def __str__(self):
      return "Classifier" + self.name + "\nParameters" + str(self.base_classifier.get_params)
   

    def fit(self,X,Y):
      unlabelled = Y == "Unlabelled"
      labelled = ~unlabelled
       


      self.partition = [Partition(sklearn.base.clone(self.base_classifier),X[labelled],Y[labelled]) for i in range(3)]
      rotations  = [rotate(self.partition,i) for (i,_) in enumerate(self.partition)]
      

      changed = True
      while changed:
        changed = False
        for t1,t2,t3 in rotations:
          changed |= t1.train(t2,t3,X[unlabelled]) 

    def predict(self,X):
      Predicted = np.asarray([third.predict(X) for third in self.partition])
      return scipy.stats.mstats.mode(Predicted.astype(int)).mode[0]
  

    def score(self,X,Y):
      y_original = Y.astype(int)
      y_pred = self.predict(X)
      return accuracy_score(y_original,y_pred)

class Partition:
    def __init__(self,classifier,La_x,La_y):
      self.classifier = classifier
      self.La_x  = La_x
      self.La_y  = La_y
      sample = sklearn.utils.resample(self.La_x,self.La_y)
      self.classifier.fit(*sample)
      self.err_prime = 0.5
      self.l_prime   = 0.0
    def update(self,L_x,L_y,error):
     print(self.La_x)
     print(L_x)
     X = np.append(self.La_x,L_x,axis=0)
     Y = np.append(self.La_y,L_y,axis=0)
     self.classifier.fit(X,Y)
     self.err_prime = error
     self.l_prime = len(L_x)
    def train(self,t1,t2,UnLa_x):
     L_x = []
     L_y  =[]
     error = self.error(t1,t2)
     if(error>=self.err_prime):
       return False
     
     for X in UnLa_x:
       X = X.reshape(1,-1)
       Y = t1.predict(X)
       if(Y == t2.predict(X)):
         L_x.append(X)
         L_y.append(Y)

     count_add = len(L_x)
     L_x = np.concatenate(L_x)
     L_y = np.concatenate(L_y)
     if(self.l_prime==0):
       self.l_prime = floor(error/(self.err_prime-error)+1)

     if(self.l_prime>=count_add):
       return False
     
     if error*count_add<self.err_prime*self.l_prime:
       self.update(L_x,L_y,error)
       return True
 
     if self.l_prime > error/(self.err_prime-error):
       n = floor(self.err_prime*self.l_prime/error-1)
       L_x,L_y = sklearn.utils.resample(L_x,L_y,replace=False,n_samples=n)
       self.update(L_x,L_y,error)
       return True
     return False
  

    def error(self,t1,t2):
     pred_1 = t1.predict(self.La_x)
     pred_2 = t2.predict(self.La_x)
     both_incorrect = np.count_nonzero((pred_1 != self.La_y)&(pred_2 != self.La_y))
     both_same = np.count_nonzero(pred_1 == pred_2)
     if(both_same == 0): 
       return np.inf
     error = both_incorrect/both_same
     return error
  

    def predict(self,*args,**kwargs):
     return self.classifier.predict(*args,**kwargs) 
def rotate(l,n):
  return l[n:]+l[:n]    