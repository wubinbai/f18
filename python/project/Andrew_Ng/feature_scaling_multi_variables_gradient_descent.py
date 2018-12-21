from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming y = 0 + 10 * x1 + 20*x2
# So theta0 = 0, theta1 = 10, theta2 = 20
x1=[1,2,3,4,5]
x2=[2000,4000,6000,8000,10000]

x1=[i/max(x1) for i in x1]
x2=[i/max(x2) for i in x2]

y=[]
for i in range(len(x1)):
    y.append( 10 * x1[i] + 20 * x2[i])

theta0=int(input('Please input a number for theta0: '))
theta1=int(input('Please input a number for theta1: '))
theta2=int(input('Please input a number for theta2: '))


def h(x1,x2):
    global theta0
    global theta1
    global theta2
    y=theta0+theta1*x1+theta2*x2
    return y
n = 50000 # iterate over n times
number = 0
alpha = 0.01
while number < n:
    h_val = []
    for i in range(len(x1)):
        h_val.append(h(x1[i],x2[i]))
    # h_val=[h(i) for i in x]
    h_minus_y=[]
    for i in range(len(h_val)):
        h_minus_y.append(h_val[i]-y[i])
    temp0=theta0-alpha*sum(h_minus_y)/len(x1)
    temp1=theta1-alpha*sum(np.array(h_minus_y) * np.array(x1)) /len(x1)
    temp2=theta2-alpha*sum(np.array(h_minus_y) * np.array(x2)) /len(x2)
    
    theta0=temp0
    theta1=temp1
    theta2=temp2
    print('number: ',number)
    number+=1
print('theta0: ', theta0)
print('theta1: ', theta1)
print('theta2: ', theta2)


mse_half=mean_squared_error(h_val,y)/2
print('mse_half: ', mse_half)
