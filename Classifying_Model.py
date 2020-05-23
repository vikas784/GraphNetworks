import numpy as np
import pandas as pd
import networkx as nx
from SDNE import SDNE
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from Main_training import Training
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt



class Classifying_Model:
  

  def __init__(self,Labelled_embedding,Train_label,Unlabelled_embedding,Test_label):
    self.Labelled_embedding = Labelled_embedding
    self.Unlabelled_embedding = Unlabelled_embedding
    self.Train_label = Train_label
    self.Test_label = Test_label
    self.Train_rmse = []
    self.Test_rmse = []
    self.results = None
    self.Data = {}
    self.i=0
    self.cnf_matrixes = {}
    self.Unlabelled_embedding = Unlabelled_embedding
    self.Data["Encoded"] = {
                   "X": self.Labelled_embedding,
                   "Y": self.Train_label
                  }

    self.KNN = KNeighborsClassifier(
                                    n_neighbors = 3,
                                    metric = "minkowski" 
                                   )
    self.CART = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=3,
    )
    self.Classifiers = [
                         Training("KNN",self.KNN),
                         Training("KNN",self.CART)                    
                        ]
    self.Labelling_rates = [0.01,0.05,0.10,0.2,0.3,0.4]
    self.Process()
  
  def Process2(self,classifiers,X,Y,cv,rate,Unlabelled_embedding,Test_label):
    Transductive_scores = []
    Testing_scores = []
    for Tr_idx,Te_idx in cv.split(X,Y):
      Transductive_score,Testing_score,cnf_matrix = self.Process3(classifiers,X,Y,Tr_idx,Te_idx,rate,Unlabelled_embedding,Test_label)
      Transductive_scores.append(Transductive_score)
      Testing_scores.append(Testing_score)
      print("#",end=" ")
    print()
    scores = {
              "trans_mean": np.mean(Transductive_scores),
              "testing_mean" :np.mean(Testing_scores),
              "trans_std": np.std(Transductive_scores),
              "testing_std" :np.std(Testing_scores),  
              }
    return scores,cnf_matrix
  
  def rmse(self,predictions, targets):
      return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())
  
  def Process3(self,classifiers,X,Y,Tr_idx,Te_idx,rate,Unlabelled_embedding,Test_label):
    Test_x = np.array(X)[Te_idx]
    Test_y = np.array(Y)[Te_idx]
    Train_Test_split = train_test_split(np.array(X)[Tr_idx],np.array(Y)[Tr_idx],test_size = rate,random_state = 42)
    (X_unlabelled,X_labelled,Y_unlabelled,Y_labelled) = Train_Test_split
    X_train = np.concatenate((X_labelled,X_unlabelled))
    Y_train = np.concatenate((Y_labelled.astype(str),np.full_like(Y_unlabelled.astype(str),"Unlabelled")))
    classifiers.fit(X_train,Y_train)
    

    transductive_score = classifiers.score(X_unlabelled,Y_unlabelled)
    testing_score = classifiers.score(Test_x,Test_y)
    Main_Unlabelled = classifiers.predict(Unlabelled_embedding)
    Te_Rmse = self.rmse(Main_Unlabelled,Test_label)
    Tr_Rmse = self.rmse(Test_y,classifiers.predict(Test_x))
    error = []
    a = []
    for i in range(len(Test_label)):
      if(Main_Unlabelled[i]==Test_label[i]):
        a.append(1)
      else:
        a.append(0)
    print("error",np.mean(a)*100)
    self.Test_rmse.append(Te_Rmse)
    plt.plot(self.Test_rmse)
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    print("Train_Rmse",Tr_Rmse,'\t',"Te_Rmse",Te_Rmse)
    
    cnf_matrix = pd.DataFrame(Test_y.astype(str),classifiers.predict(Test_x))
    return transductive_score,testing_score,cnf_matrix

  def Process(self):
    
    for classifiers in self.Classifiers:
      self.cnf_matrixes[classifiers.name] = {}
      print(classifiers.name)
      for Name,dataset in self.Data.items():
        self.cnf_matrixes[classifiers.name][Name] = {}
        print("Dataset:",Name,"\t")
        for rate in self.Labelling_rates:
          print("Labelling_rate:",rate,end=" ")
          test_info = {"Classifier":classifiers.name,"Dataset":Name,"Labelling_Rate":rate}
          cv = KFold(n_splits=15,random_state=50)
          scores,cnf_matrix = self.Process2(classifiers,self.Data['Encoded']["X"],self.Data['Encoded']["Y"],cv,rate,self.Unlabelled_embedding,self.Test_label)
          if self.results is None:
            self.results = pd.DataFrame([{**test_info,**scores}])
          else:
            self.results.loc[len(self.results.index)] = {**test_info,**scores}
          self.cnf_matrixes[classifiers.name][Name][rate] = cnf_matrix

    print()
    print("--------")
    print(self.results)
          
        
      
    