from __future__ import division  # floating point division
import numpy as np
import utilities as utils
from random import randrange


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))
    
    
class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'usecolumnones': False}
        self.reset(parameters)
            
    def reset(self, parameters):
        self.resetparams(parameters)
        # TODO: set up required variables for learning
    
    def learn(self, X, y):
        if self.params['usecolumnones'] == False:
            X = X[:,:-1]
        self.para_c = [(X[y==ytype].mean(axis=0), X[y==ytype].var(axis=0)) for ytype in np.unique(y)]
        self.prob_y = [sum(y==ytype)/len(y) for ytype in np.unique(y)]
        
    def self_mult(self, lst):
        prod = 1
        for i in lst:
            if np.isnan(i):
                prod = prod*1
            else:
                prod = prod*i
        return prod
    
    def gauss(self, mu, var, x):
        if var == 0:
            if np.abs(x-mu) < 1e-2:
                return 1.0
            else:
                return 0
        else:
            return ((1/np.sqrt(2*np.pi*var))*np.exp((x-mu)**2/(-2*var)))
    def gauss_pdf(self, mu, var, x):
        return ((1/np.sqrt(2*np.pi*var))*np.exp((x-mu)**2/(-2*var)))
    
    def predict(self, X):
        if self.params['usecolumnones'] == False:
            X = X[:,:-1]
        prob1 = np.array([self.self_mult(row) for row in self.gauss_pdf(self.para_c[0][0], self.para_c[0][1], X)])*self.prob_y[0]
        prob0 = np.array([self.self_mult(row) for row in self.gauss_pdf(self.para_c[1][0], self.para_c[1][1], X)])*self.prob_y[1]
        return np.array([0 if x>0 else 1 for x in prob1-prob0])
    # TODO: implement learn and predict functions                  
            
class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'lmbda1': .01, 'lmbda2': .1}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = 'l1'
            self.regwgt = self.params['regwgt']
        elif self.params['regularizer'] is 'l2':
            self.regularizer = 'l2'
            #print(self.params['regwgt'])
            self.regwgt = self.params['regwgt']
        elif self.params['regularizer'] is 'elasticNet':
            self.regularizer = 'elasticNet'
            self.lmbda1 = .01
            self.lmbda2 = .1
        else:
            self.regularizer = 'basic'
    
    def prox_func(self, delE, lmd):
        return np.array([(x-lmd) if x>lmd else x+lmd if x<-lmd else 0 for x in delE])    
    
    def learn(self, X,y):
        w = np.array([randrange(-10,10) for i in range(0,X.shape[1])])

        epoch_error = []
        alpha = .1
        #self.regwgt = self.params['regwgt']
        for epoch in range(0,500):
            
            small_p = utils.sigmoid(np.dot(X,w))
            err_old = geterror([1 if x>.5 else 0 for x in utils.sigmoid(np.dot(X,w))],y)
            if self.regularizer == 'basic':
                w = w - alpha*(X.T.dot(np.subtract(small_p,y)))
            elif self.regularizer == 'l2':
                w = w - alpha*(X.T.dot(np.subtract(small_p,y))+ 2*self.regwgt*w)
            elif self.regularizer == 'l1':
                w = self.prox_func(w - alpha*(X.T.dot(np.subtract(small_p,y))), self.regwgt)
            elif self.regularizer == 'elasticNet':
                w = self.prox_func(w - alpha*(X.T.dot(np.subtract(small_p,y)) + ((self.lmbda1*(1-self.lmbda2)))*w), self.lmbda1*self.lmbda2)   
           
            err_new = geterror([1 if x>.5 else 0 for x in utils.sigmoid(np.dot(X,w))],y)
            if err_new>err_old:
                alpha = alpha*.1
                
            epoch_error.append(err_new)
        self.weights = w

    def predict(self, X):
        return np.array([1 if x>.5 else 0 for x in utils.sigmoid(np.dot(X,self.weights))])
    # TODO: implement learn and predict functions                  
           

class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                        'transfer': 'sigmoid',
                        'stepsize': 0.01,
                        'epochs': 10}
        self.reset(parameters)        
        
    def sigmoid(self, xvec):
    
        return 1.0 / (1.0 + np.exp(np.negative(xvec)))
    
    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')      
        self.wi = None
        self.wo = None
    
    def learn(self, X, y):
        hls = self.params['nh']
        w2 = np.zeros(shape=(X.shape[1], hls))
        w1 = np.array([randrange(-100,100) for i in range(0,hls)])
        alpha = .001
        
        
        for epoch in range(0,self.params['epochs']):
            state = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(state)
            np.random.shuffle(y) 
            for t in range(0, np.shape(X)[0]):
                
                y2 = utils.sigmoid(np.dot(X[t],w2))
                y1 = self.sigmoid(np.dot(y2,w1.T))
                del1 = y1-y[t]
                grad1 = np.dot(del1.T,y2)
                del2 = np.array([(w1*del1)[i]*y2[i]*(1-y2[i]) for i in range(len(w1))])
                grad2 = np.array([X[t]*i for i in del2]).T
                
                w2 = w2-alpha*grad2
                w1 = w1-alpha*grad1
        self.wi = w2
        self.wo = w1
    # TODO: implement learn and predict functions                  

    def predict(self, X):
        y2 = utils.sigmoid(np.dot(X,self.wi))
        y1 = utils.sigmoid(np.dot(y2,self.wo.T))  
        return np.array([1 if x>.5 else 0 for x in y1])
        
    def _evaluate(self, inputs):
        """ 
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)


class LogitRegAlternative(Classifier):

    def __init__( self, parameters={} ):
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
    
    def pred(self, x):
        a = np.sqrt(1+x**2)
        return .5*(1+(x/a))
    
    def gradient(self, x,y,w):
        a = np.sqrt(1+np.dot(x,w)**2)
        b = (x/a)*(y-self.pred(np.dot(x,w)))
        return 2*b
        
    def learn(self, X, y):    
        w = np.array([randrange(-10,10) for i in range(0,X.shape[1])])
        alpha = .1
        for epoch in range(0,100):
            err_old = geterror([1 if x>.5 else 0 for x in self.pred(np.dot(X,w))],y)
            grad = sum([self.gradient(X[i],y[i],w) for i in range(y.shape[0])])
            w = w + alpha*grad
            err_new = geterror([1 if x>.5 else 0 for x in self.pred(np.dot(X,w))],y)
            
            if err_new>err_old:
                alpha = alpha*.1
        
        self.weights = w

    
    def predict(self, X):
        return np.array([1 if x>.5 else 0 for x in self.pred(np.dot(X,self.weights))])
        
    # TODO: implement learn and predict functions                  
           
    
