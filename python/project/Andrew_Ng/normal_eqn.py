# X * theta = Y 
""" README:
4-6 Normal Eqn.: (no need to use feature scaling method) theta = (X^TX)^-1X^Ty, my own very simple derivation is as follows: 3 eqns:
1. X*theta = y (X is m by n+1, m: # of observations/samples, n: # features, theta is n+1 by 1, y is m by 1)
2. X^TX*theta=X^Ty
3. theta = (X^TX)^-1X^Ty
"""

import numpy as np
# Assuming y = 0 + 10 * x1 + 20*x2
# So theta0 = 0, theta1 = 10, theta2 = 20
x1=[1,2,3,4,5]
x2=[2,4,6,8,10]
y=[]
for i in range(len(x1)):
    y.append( 10 * x1[i] + 20 * x2[i])

X=np.matrix([[1]*len(x1),x1,x2]).T
y=np.matrix(y).T
theta = np.linalg.pinv(X.T*X)*X.T*y
