
"""
Groupby technique
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=True).mean().sort_values(by='Survived',ascending=False)

You can do this with Sex, SibSp, Parch.

Visualization:
g=sns.FacetGrid(train_df_copy,col='Survived')
g.map(plt.hist,'Age',bins=20)

grid=sns.FacetGrid(train_df_copy,col='Survived',row='Pclass')
grid.map(plt.hist,'Age',alpha=.5,bins=20)

grid2=sns.FacetGrid(train_df_copy,row='Embarked',height=2.2,aspect=1.6)
grid2.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')
GRID2.ADD_LEGEND()

grid3=sns.FacetGrid(train_df_copy,row='Embarked',col='Survived',height=2.2,aspect=1.6)
grid3.map(sns.barplot,'Sex','Fare',alpha=.5,ci=None)


ccombo2=[train_df_copy,test_df_copy]
for i in combo2:
     i['Title'] = i.Name.str.extract('([A-Za-z]+)\.',expand=False)

pd.crosstab(train_df_copy.Title,train_df_copy.Sex)

grid4 = sns.FacetGrid(train_df_copy,row='Pclass',col='Sex',size=2.2,aspect=1.6)
grid4.map(plt.hist,'Age',alpha=.5,bins=20) 

train_df_copy['AgeBand']=pd.cut(train_df_copy['Age'],5) 

train_df_copy[['AgeBand','Survived']].groupby(['AgeBand'],as
    ...: _index=False).mean() 


"""







import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# my import
# from sklearn.metrics import accuracy_score

# Step 1: Import and Modify data

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
train_df_copy = train_df
test_df_copy = test_df
combine=[train_df,test_df]
PassengerId_Series = test_df["PassengerId"]

'''
columns=train_df.columns
columns
columns_list=list(columns)
columns_list
'''

'''
d=dict()
func_list=['Ticket, Cabin','Name, PassengerId','Sex','Age','SibSp, Parch','Age*Class','Embarked','Fare']
for i in range(8):
    d['func'+'{}'.format(i+1)]=func_list[i]
'''
# define func1 to func8

def func1():
    '''func1: drop Ticket & Cabin'''
    global train_df
    global test_df
    global combine
    # Drop Ticket since by using train_df_copy.Ticket.describe() we know it's highly duplicated, with 681 unique, out of 891 in training set. More importantly, there may not be correlation between Ticket and Survived.
    train_df=train_df.drop('Ticket',axis = 1)
    test_df=test_df.drop('Ticket', axis = 1)
    # Drop Cabin since it's highly incomplete/ contains too many null values. 204/891; 91/418.
    train_df=train_df.drop('Cabin',axis = 1)
    test_df=test_df.drop('Cabin', axis = 1)
    combine=[train_df,test_df]

def func2():
    '''func2: drop PassengerId & Name'''
    global train_df
    global test_df
    global combine
    #For test_df, do not drop PassengerId yet.
    train_df=train_df.drop('PassengerId',axis = 1)
    #test_df=test_df.drop('PassengerId', axis = 1)
    train_df=train_df.drop('Name',axis = 1)
    test_df=test_df.drop('Name', axis = 1)
    combine=[train_df,test_df]


def func3():
    '''func3: Sex: male 0; female 1'''
    global train_df
    global test_df
    global combine
    
    mapping = {'male':0,'female':1}
    train_df['Sex'] = train_df['Sex'].map(mapping).astype(int)
    test_df['Sex'] = test_df['Sex'].map(mapping).astype(int)
    combine=[train_df,test_df]

# LEAF
   
def func4():
    '''func4: fill NaN for age with median for both train and test data  and divide age into 5 categories 0-4, with 16-year increment'''
    global train_df
    global test_df
    global combine

    train_df['Age'].fillna(train_df['Age'].dropna().median(), inplace = True)
    test_df['Age'].fillna(test_df['Age'].dropna().median(), inplace = True)
    for i in combine:
        i.loc[i['Age'] <= 16,'Age'] = 0
        i.loc[(i['Age'] > 16) & (i['Age'] <= 32),'Age'] = 1
        i.loc[(i['Age'] > 32) & (i['Age'] <= 48),'Age'] = 2
        i.loc[(i['Age'] > 48) & (i['Age'] <= 64),'Age'] = 3
        i.loc[(i['Age'] > 64),'Age'] = 4

# LEAF B

def func5():
    '''Create FamilySize from SibSp and Parch then create IsAlone'''
    global train_df
    global test_df
    global combine

    for i in combine:
        i['FamilySize'] = i['SibSp'] + i['Parch'] + 1
    for i in combine:
        i['IsAlone'] = 0
        i.loc[i['FamilySize']==1,'IsAlone'] = 1
    train_df=train_df.drop(['Parch', 'SibSp','FamilySize'], axis = 1)
    test_df=test_df.drop(['Parch', 'SibSp','FamilySize'], axis = 1)
    combine = [train_df,test_df]

def func6():
    '''Create Age*Class'''
    global train_df
    global test_df
    global combine

    for i in combine:
        i['Age*Class'] = i.Age * i.Pclass

def func7():
    '''fill NaN for missing data in train data's Embarked with mode, then convert to 0 1 2'''
    global train_df
    global test_df
    global combine

    for i in combine:
        i['Embarked'] = i['Embarked'].fillna(train_df.Embarked.dropna().mode()[0])# since only train data's Embarked has nan
        i['Embarked'] = i['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)


def func8():
    '''fill with median for one missing in test data's Fare, then catogorize into 4 section with 0-3'''
    global train_df
    global test_df
    global combine

    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace = True)
    for i in combine:
        i.loc[i['Fare'] <= 7.91,'Fare'] = 0
        i.loc[(i['Fare'] > 7.91) & (i['Fare'] <= 14.454),'Fare'] = 1
        i.loc[(i['Fare'] > 14.454) & (i['Fare'] <= 31),'Fare'] = 2
        i.loc[(i['Fare'] > 31),'Fare'] = 3
    
    combine = [train_df, test_df]



func1()
func2()
func3()
func4()
func5()
func6()
func7()
func8()



# Step 2: use data
# overall:
X_train = train_df.drop("Survived", axis = 1)
y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis = 1)
# ML models: LR, SVM, kNN, GNB, Perceptron, Linear SVC, SGD, DT, RF



# 1 LR
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train)*100,2)
y_pred_lr = y_pred
sub_lr = pd.DataFrame({"PassengerId":PassengerId_Series,"Survived":y_pred_lr})
sub_lr.to_csv('sub_lr.csv', index = False)

from sklearn.model_selection import cross_val_score as crvs
logreg_score = crvs(logreg,X_train,y_train,cv=100,scoring='accuracy')




# 2 SVM
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train)*100,2)
y_pred_svc = y_pred
sub_svc = pd.DataFrame({"PassengerId":PassengerId_Series,"Survived":y_pred_svc})
sub_svc.to_csv('sub_svc.csv', index = False)


# 3 kNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train)*100,2)
y_pred_knn = y_pred
sub_knn = pd.DataFrame({"PassengerId":PassengerId_Series,"Survived":y_pred_knn})
sub_knn.to_csv('sub_knn.csv', index = False)



# 4 GNB
gau = GaussianNB()
gau.fit(X_train, y_train)
y_pred = gau.predict(X_test)
acc_gau = round(gau.score(X_train, y_train)*100,2)
y_pred_gnb = y_pred
sub_gnb = pd.DataFrame({"PassengerId":PassengerId_Series,"Survived":y_pred_gnb})
sub_gnb.to_csv('sub_gnb.csv', index = False)



# 5 Perceptron
perc = Perceptron()
perc.fit(X_train, y_train)
y_pred = perc.predict(X_test)
acc_perc = round(perc.score(X_train, y_train)*100,2)
y_pred_perc = y_pred
sub_perc = pd.DataFrame({"PassengerId":PassengerId_Series,"Survived":y_pred_perc})
sub_perc.to_csv('sub_perc.csv', index = False)



# 6 Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train)*100,2)
y_pred_lsvc = y_pred
sub_lsvc = pd.DataFrame({"PassengerId":PassengerId_Series,"Survived":y_pred_lsvc})
sub_lsvc.to_csv('sub_lsvc.csv', index = False)


# 7 SGD
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, y_train)*100,2)
y_pred_sgd = y_pred
sub_sgd = pd.DataFrame({"PassengerId":PassengerId_Series,"Survived":y_pred_sgd})
sub_sgd.to_csv('sub_sgd.csv', index = False)



# 8 DT
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
acc_dt = round(dt.score(X_train, y_train)*100,2)
y_pred_dt = y_pred
sub_dt = pd.DataFrame({"PassengerId":PassengerId_Series,"Survived":y_pred_dt})
sub_dt.to_csv('sub_dt.csv', index = False)

# 9 RF
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc_rf = round(rf.score(X_train, y_train)*100,2)
y_pred_rf = y_pred
sub_rf = pd.DataFrame({"PassengerId":PassengerId_Series,"Survived":y_pred_rf})
sub_rf.to_csv('sub_rf.csv', index = False)

rf_score = crvs(rf,X_train,y_train,cv=100,scoring='accuracy')





print(train_df.shape,test_df.shape,train_df.info(),test_df.info())
