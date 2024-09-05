import numpy as np

class Pca:
    def __init__(self, k):
        self.n_componenets = k
        self.mean = None 

    def get_pcomp(self,X):
        self.X= X
  
        #covariance matrix of standardized data
        self.cov = np.cov(self.X.T)

        #eigen values and eign vectors of covariance matrix
        self.eigenvalues,self.eigenvectors =np.linalg.eigh(self.cov)

        #Shorting eigen values in decending order with thier respective eigen values
        idX = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idX]
        self.eigenvectors = self.eigenvectors[:,idX]

        #Construction of transform or feature matrix from first n components
        self.components = self.eigenvectors[:,0:self.n_componenets]

    def transform(self,X):
        #Get projected data set 
        self.X= X
        return np.dot(self.X, self.components)

    





    
