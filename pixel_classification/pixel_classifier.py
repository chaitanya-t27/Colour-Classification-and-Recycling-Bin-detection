'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    self.mu = np.zeros((3,3))
    self.var = np.zeros((3,3))
    self.theta = np.zeros((3,1))
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    ##Using Gaussian Naive-Bayes model for pixel classification
    ##Using Gaussian Naive-Bayes model for pixel classification
    self.mu = np.array([[0.75250609, 0.34808562, 0.34891229],
       [0.35060917, 0.73551489, 0.32949353],
       [0.34735903, 0.33111351, 0.73526495]])
    
    self.var = np.array([[0.03705927, 0.06196869, 0.06202255],
       [0.05573463, 0.03478593, 0.05602188],
       [0.05453762, 0.05683331, 0.03574061]])
    
    self.theta = np.array([0.36599891716296695, 0.3245804006497022, 0.3094206821873308])
   
    
   
    def pdfunc(x, mu, var):
        return np.exp((-1*(x-mu)**2)/(2*var))*(1/(2*3.14*var)**0.5)
    

    maxap = np.zeros((3,1))
    img = X
    y = np.zeros(img.shape[0])
    for k in range(img.shape[0]):    
        for i in range(3): # i is for class number
            prob = 1
            for j in range(3): # j is for r/g/b
                prob *= pdfunc(img[k][j], self.mu[i][j], self.var[i][j])
            maxap[i] = prob * self.theta[i]
        out = np.argmax(maxap) + 1
        y[k] = (out)
        

    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

def train(self,X, y):
    """
    Function to train the Gaussian Parameters
    """
    y1_idx = np.where(y == 1)
    y2_idx = np.where(y == 2)
    y3_idx = np.where(y == 3)
    
    X1 = X[y1_idx]
    X2 = X[y2_idx]
    X3 = X[y3_idx]
    
    ##Using Gaussian Naive-Bayes model for pixel classification
    n1, n2, n3 = len(y1_idx), len(y2_idx), len(y3_idx)
    total_n = (n1 + n2 + n3)
    theta1, theta2, theta3 = n1 / total_n, n2 / total_n, n3 / total_n
    self.theta = [theta1, theta2, theta3]
    
    mu1, mu2, mu3 = np.sum(X1, axis=0) / n1, np.sum(X2, axis=0) / n2, np.sum(X3, axis=0) / n3
    self.mu = np.stack((mu1,mu2,mu3))
    
    var1 = np.sum(np.square(X1-mu1), axis=0) / n1
    var2 = np.sum(np.square(X2-mu2), axis=0) / n2                               
    var3 = np.sum(np.square(X3-mu3), axis=0) / n3
    self.var = np.stack((var1,var2,var3))
