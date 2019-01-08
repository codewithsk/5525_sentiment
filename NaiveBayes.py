import sys

import numpy as np
from Eval import Eval

from imdb import IMDBdata

class NaiveBayes:
    def __init__(self, X, Y, ALPHA=1.0):
        self.ALPHA=ALPHA
        #TODO: Initalize parameters
        self.Train(X,Y)

    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        return

    def Predict(self, X):
        #TODO: Implement Naive Bayes Classification
        return 1

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()

if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    test  = IMDBdata("%s/test" % sys.argv[1], vocab=train.vocab)
    
    nb = NaiveBayes(train.X, train.Y, float(sys.argv[2]))
    print nb.Eval(test.X, test.Y)
