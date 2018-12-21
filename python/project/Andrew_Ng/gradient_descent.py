from sklearn.metrics import mean_squared_error
import numpy as np

x=[1,2,3,4,5]
y=[10,20,30,40,50]
theta0=int(input('Please input a number for theta0: '))
theta1=int(input('Please input a number for theta1: '))
def h(x):
    global theta0
    global theta1
    y=theta0+theta1*x
    return y
n = 1000 # iterate over n times
number = 0
alpha = 0.1
while number < n:
    h_val=[h(i) for i in x]
    h_minus_y=[]
    for i in range(len(h_val)):
        h_minus_y.append(h_val[i]-y[i])
    temp0=theta0-alpha*sum(h_minus_y)/len(x)
    temp1=theta1-alpha*sum(np.array(h_minus_y) * np.array(x)) /len(x)
    theta0=temp0
    theta1=temp1
    print('number: ',number)
    number+=1
print('theta0: ', theta0)
print('theta1: ', theta1)
mse_half=mean_squared_error(h_val,y)/2
print('mse_half: ', mse_half)
