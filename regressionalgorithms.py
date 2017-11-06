from __future__ import division  # floating point division
import numpy as np
import math

import utilities as utils


def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))
    
def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """
    
    def __init__( self, params={}):
        """ Params can contain any useful parameters for the algorithm """
        self.weights = None
        self.params = {}
        
    def reset(self, params):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,params)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
        # Could also add re-initialization of weights, so that does not use previously learned weights
        # However, current learn always initializes the weights, so we will not worry about that
        
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """        
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """
    
    def __init__( self, params={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.min = 0
        self.max = 1
        self.params = {}
                
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest
        
class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, params={} ):
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)
        
    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean

        

class FS_SC_LinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5]}
        self.reset(params)    

    def stoch_descent(self, x, y, w):
        alpha = .01
        self.epoch_error = []
        for epoch in range(0,9):
            self.epoch_error.append(np.linalg.norm(np.dot(x,w) - y)/np.shape(x)[0])
            state = np.random.get_state()
            np.random.shuffle(x)
            np.random.set_state(state)
            np.random.shuffle(y)
            for t in range(0, np.shape(x)[0]):
                alpha = alpha
                w = w - alpha * np.dot((np.dot(x[t,:],w) - y[t]),x[t,:])
        return w
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        w0 = np.array(list(range(0, np.shape(Xless)[1])))
        self.weights = self.stoch_descent(Xless, ytrain, w0)
                
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest


class FS_B_LinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5]}
        self.reset(params)    

    def batch_descent(self, x, y, w):
        alpha = 1
        self.epoch_error = []
        new_error = 3
        Ew = 1
        while np.abs(new_error-Ew)>0.01:
            Ew = np.sum((np.dot(x, w) - y)**2)/x.shape[0]
            new_w = w-alpha*np.dot(x.T,(np.dot(x,w)-y))
            new_error = np.sum((np.dot(x, new_w) - y)**2)/x.shape[0]
            while new_error>=Ew:
                alpha = .5*alpha
                new_w = w-alpha*np.dot(x.T,(np.dot(x,w)-y))
                new_error = np.sum((np.dot(x, new_w) - y)**2)/x.shape[0]
            w = w-alpha*np.dot(x.T,(np.dot(x,w)-y))
            new_error = np.sum((np.dot(x, w) - y)**2)/x.shape[0]
            self.epoch_error.append(np.linalg.norm(np.dot(x,w) - y)/np.shape(x)[0])
        return(w)
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        w0 = np.array(list(range(0, np.shape(Xless)[1])))
        self.weights = self.batch_descent(Xless, ytrain, w0)
                
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest



class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5]}
        self.reset(params)    
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest


class FS_pinv_LinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5]}
        self.reset(params)    
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xless.T,Xless)), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest

class FSRidgeRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5], 'lambda': 0}
        self.reset(params)  
        #self.lambda = lambda
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)+(self.params['lambda']*np.identity(np.shape(Xless)[1]))), Xless.T),ytrain)
        
    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]        
        ytest = np.dot(Xless, self.weights)       
        return ytest



class MPLinearRegression(Regressor):
    """
    Linear Regression with feature selection
    """
    def __init__( self, params={} ):
        self.weights = None
        self.params = {'features': [1,2,3,4,5]}
        self.reset(params)  
        #self.lambda = lambda
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        xless = Xtrain[:,self.params['features']]
        #Starting with adding constant feature
        x = np.ones(np.shape(xless)[0])

        self.weights = np.array([np.sum(ytrain)/np.shape(ytrain)[0]])
        
        residuals = ytrain - x*self.weights
        
        self.select_cols = []
        self.highest_corr = []
        self.errors_for = []
        self.max_col_corr = 1
        while self.max_col_corr>=.05:
            col_corr = [np.abs(np.corrcoef(residuals,col)[0,1]) for col in xless.T]
            self.max_col_corr = max(col_corr)
            highest_corr_index = col_corr.index(max(col_corr))
            self.select_cols.append(highest_corr_index)
            self.highest_corr.append(max(col_corr))
            x = np.column_stack((x, xless[:,highest_corr_index]))
            self.weights = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)), x.T),ytrain)
            
            self.errors_for.append(geterror(np.dot(x,self.weights),ytrain))
            #print(geterror(np.dot(x,weights),ytrain))
            residuals = ytrain - np.dot(x,self.weights)
        
        
    def predict(self, Xtest):
        xless = Xtest[:,self.params['features']]
        x = np.ones(np.shape(xless)[0])
        x = np.column_stack((x, xless[:,self.select_cols]))
        ytest = np.dot(x, self.weights)       
        return ytest