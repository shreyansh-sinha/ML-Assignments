import numpy as np 
#import pandas as pd 
import matplotlib.pyplot as plt

def hypothesis(theta, X, n): # h = X.B_transpose
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
      h[i] = float(np.matmul(theta, X[i]))
      h = h.reshape(X.shape[0])
    return h

# iterative updation

def gradient_descent(theta, learning_rate, iterations, h, X, Y, n):
	cost = np.ones(iterations)

	for i in range(0, iterations):
		theta[0] = theta[0] - (learning_rate/X.shape[0]) * sum(h - Y)

		for j in range(1, n+1):
			theta[j] = theta[j] - (learning_rate/X.shape[0]) * sum((h - Y) * X.transpose()[j])

		h = hypothesis(theta, X, n)

		# cost function = 1/(2*m) (sigma(h(x) - y) ** 2)
		cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - Y))

	theta = theta.reshape(1, n+1)
	return theta, cost


def linear_regression(X, y, alpha, num_iters):
    n = X.shape[1] #size of X
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    # initializing the parameter vector...
    theta = np.zeros(n+1)
    #print(theta) 
    # hypothesis calculation....
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent...
    theta, cost = gradient_descent(theta,alpha,num_iters,h,X,y,n)
    return theta, cost


data = np.loadtxt('airfoil_self_noise.dat', delimiter='\t')
X_train = data[:,:-1] #feature set...select all the input values
y_train = data[:,5] #label set...select the output values

mean = np.ones(X_train.shape[1]) # define mean array
std_dev = np.ones(X_train.shape[1]) # define standard deviation array


# Scaling Data
# shape attriute for numpy arrays returns dimensions orf array
# if X has n rows and m columns then X.shape[0] is n and X.shape[1]
# is m

for i in range(0, X_train.shape[1]):
    mean[i] = np.mean(X_train.transpose()[i])
    std_dev[i] = np.std(X_train.transpose()[i])
    for j in range(0, X_train.shape[0]):
        X_train[j][i] = (X_train[j][i] - mean[i])/std_dev[i]


iterations = 10000
learning_rate = 0.005
theta, cost = linear_regression(X_train, y_train, learning_rate, iterations)

print(theta)
print(cost[iterations-1])

cost = list(cost)
n_iterations = [x for x in range(1, 10001)]
plt.plot(n_iterations, cost)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost Value')
