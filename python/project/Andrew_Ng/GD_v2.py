import numpy as np

def compute_cost(X,y,theta):
    m = y.size
    cost = np.sum((np.dot(X,theta)-y)**2)/(2*m)
    return cost

# e.g.
X = np.array([[1,1,1.8],[1,2,4],[1,3,6],[1,4,9],[1,5,10.5]])
y = np.array([40,100,150,220,260]).T
theta = np.array([0,2,6]).T
theta = theta.astype('float64')
cost = compute_cost(X,y,theta)
print(cost)

def GD(X,y,theta,alpha,num_iters):
    m=y.size
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        err = np.dot(X,theta)-y
        theta -= (alpha/m) * np.dot(err.T,X).T
        J_history[i] = compute_cost(X,y,theta)
    return theta, J_history


alpha = 0.01
num_iters = 3000
ans1, ans2 = GD(X,y,theta,alpha,num_iters)


