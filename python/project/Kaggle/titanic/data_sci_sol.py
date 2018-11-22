import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Step 1: Import and Modify data

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
combine=[train_df,test_df]
columns=train_df.columns
columns
columns_list=list(columns)
columns_list
d=dict()
func_list=['Ticket, Cabin','Name, PassengerId','Sex','Age','SibSp, Parch','Age*Class','Embarked','Fare']
for i in range(8):
    d['func'+'{}'.format(i+1)]=func_list[i]

# define func1 to func8

def func1():
    global train_df
    global test_df
    train_df=train_df.drop('Ticket',axis = 1)
    test_df=test_df.drop('Ticket', axis = 1)
    train_df=train_df.drop('Cabin',axis = 1)
    test_df=test_df.drop('Cabin', axis = 1)
    combine=[train_df,test_df]

def func2():
    global train_df
    global test_df
    train_df=train_df.drop('PassengerId',axis = 1)
    test_df=test_df.drop('PassengerId', axis = 1)
    train_df=train_df.drop('Name',axis = 1)
    test_df=test_df.drop('Name', axis = 1)

def func3():
    global train_df
    global test_df
    for i in range(len(train_df.Sex)):
        if train_df.Sex[i] == 'male':
            train_df.Sex[i] = 0
        else:
            train_df.Sex[i] = 1
     for i in range(len(test_df.Sex)):
        if test_df.Sex[i] == 'male':
            test_df.Sex[i] = 0
        else:
            test_df.Sex[i] = 1
    #train_df.Sex

func1()
func2()
func3()
train_df.shape
test_df.shape
