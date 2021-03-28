#importing the necessary libraries
import numpy as np
from sklearn import datasets
import math
from sklearn.metrics import accuracy_score
#creating dummy dataset for binary classification
data=datasets.make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2)
X=data[0]#features
y=data[1]#labels
print(X)
print(y)
print("shape of x: {} \n".format(X.shape))#printing the shape of x and y
print("shape of y : {} \n".format(y.shape))
n = len(X)
# Learning rate
alpha=float(input("enter the learning rate:"))
iterations=int(input("\nenter the number of iterations:"))
class LogisticRegression: #creating a class for logitsic regression
    def __init__(self,alpha,iterations,n,X,y):
        self.alpha=alpha
        self.iterations=iterations
        self.n=n
        self.X=X
        self.y=y
    def sigmoid(self,x): #defining the sigmoid function
        return 1/(1+np.exp(-x))

    def predict(self,w0,w1,w2,w3,X,b):#predict function which accepts the weights,bias term and x as paramters
       return [self.sigmoid(w0*x[0]+w1*x[1]+w2*x[2]+w3*x[3]+b) for x in X]

   

    def gradient_descent(self):#gardient descent function which accepts learning rate,ietrations,weights,bias,X and y
        #initialising the parameters
           
        lamda=[0.01,0.1,0.2,1,2]
        for l in lamda:
            w0,w1,w2,w3=0.01,0.01,0.01,0.01
            b=0.01
            for i in range(iterations):
                y_hat = self.predict(w0,w1,w2,w3,X,b)#predicting the output for X
                if w0>0:
                    delta_w0 = -np.sum([(y[j] - y_hat[j])*X[j,0] for j in range(n)])/n+ l*1#finding the gradient of  cost function  wrt each parameter
                if w0<0:
                    delta_w0 = -np.sum([(y[j] - y_hat[j])*X[j,0] for j in range(n)])/n-l*1
                if w1>0:
                    delta_w1 = -np.sum([(y[j] - y_hat[j])*X[j,1] for j in range(n)])/n+ l*1
                if w1<0:
                    delta_w1 = -np.sum([(y[j] - y_hat[j])*X[j,1] for j in range(n)])/n-l*1
                if w2>0:
                    delta_w2 = -np.sum([(y[j] - y_hat[j])*X[j,2] for j in range(n)])/n+l*1
                if w2<0:
                    delta_w2 = -np.sum([(y[j] - y_hat[j])*X[j,2] for j in range(n)])/n-l*1
                if w3>0:
                    delta_w3 = -np.sum([(y[j] - y_hat[j])*X[j,3] for j in range(n)])/n+l*1
                if w3<0:
                    delta_w3 = -np.sum([(y[j] - y_hat[j])*X[j,3] for j in range(n)])/n-l*1
                if b>0:
                    delta_b = -np.sum([(y[j] - y_hat[j]) for j in range(n)])/n + l*1
                if b<0:
                    delta_b = -np.sum([(y[j] - y_hat[j]) for j in range(n)])/n -l*1


                w0 = w0 - alpha*delta_w0#updating the weights
                w1 = w1 - alpha*delta_w1
                w2 = w2 - alpha*delta_w2
                w3 = w3 - alpha*delta_w3
                print("updated weights for iteration {} is : \n".format(i))
                print(w0,"\t",w1,"\t",w2,"\t",w3,"\n")
                b = b - alpha*delta_b
                print(b,"\n")
                
                if np.sum(np.abs([delta_w0, delta_w1, delta_w2, delta_w3, delta_b])) < 1e-5:
                    break

            # Make predictions
            pred = [1 if i > 0.5 else 0 for i in y_hat]
            # Find no:of correct predictions
            correct  = np.sum([1 if pred[i] == y[i] else 0 for i in range(n)])
            print ("number of correct predictions for lamda = {}  : \t {} ".format(l,correct))
            print("accuracy:\t",accuracy_score(y,pred))
   
lr=LogisticRegression(alpha,iterations,n,X,y)
lr.gradient_descent()#calling the gradient descent function
