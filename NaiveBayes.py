# NaiveBayes.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_spring19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import sys

import numpy as np
from Eval import Eval

from imdb import IMDBdata

import pdb

class NaiveBayes:
    def __init__(self, X, Y, ALPHA=1.0):
        self.ALPHA=ALPHA
        #TODO: Initalize parameters
        self.Train(X,Y)

    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        self.prob_pos = np.log(np.sum(Y==1.0) / float(len(Y)))
        self.prob_neg = np.log(np.sum(Y==-1.0) / float(len(Y)))
        
        filt = X #(X > 0) * 1.

        self.feat_pos = filt[Y==1.0]
        self.feat_neg = filt[Y==-1.0]
        
        self.feat_pos = ((np.sum(self.feat_pos, axis = 0)  + self.ALPHA) / (float(np.sum(self.feat_pos)) + self.ALPHA * X.shape[1]))
        self.feat_pos = np.log(self.feat_pos)

        self.feat_neg = ((np.sum(self.feat_neg, axis = 0) + self.ALPHA)  / (float(np.sum(self.feat_neg)) + self.ALPHA * X.shape[1]))
        self.feat_neg = np.log(self.feat_neg)

	
        #self.feat_pos = np.log(filt[Y == 1.0].sum(axis = 0) / float(len(Y[Y==1.0])))
        #self.feat_neg = np.log(filt[Y == -1.0].sum(axis = 0) / float(len(Y[Y==-1.0])))

        #self.feat_pos[np.isinf(self.feat_pos)] = 0.0
        #self.feat_neg[np.isinf(self.feat_neg)] = 0.0
       
        self.feat_pos = np.array(self.feat_pos).flatten()
        self.feat_neg = np.array(self.feat_neg).flatten()
 
        return

    def Predict(self, X):
        #TODO: Implement Naive Bayes Classification

        predictions = []
        i = 0
        for x in X:
            #print i
            i+=1
            pos = self.prob_pos + np.sum(np.multiply((x>0).todense(), self.feat_pos))
            neg = self.prob_neg + np.sum(np.multiply((x>0).todense(), self.feat_neg))
            if pos > neg:
                predictions.append(1.0)
            else:
                predictions.append(-1.0)
        return np.array(predictions)

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()

if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    
    nb = NaiveBayes(train.X, train.Y, float(sys.argv[2]))
    print nb.Eval(test.X, test.Y)
