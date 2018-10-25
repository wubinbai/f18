import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=5)



tr_d=pd.read_csv('train.csv')
te_d=pd.read_csv('test.csv')

#print(tr_d.head())
y_tr=tr_d[tr_d.columns[1]]
x_tr=tr_d[tr_d.columns[:1].append(tr_d.columns[2:])]

#y_te=te_d#[te_d.columns[1]]
x_te=te_d#[te_d.columns[:1].append(te_d.columns[2:])]

#print(tr_d.columns)
#print(y_tr.columns)

feature_cols = ['PassengerId','Pclass','Age','SibSp','Parch','Fare']
x_tr=x_tr[feature_cols]
x_te=x_te[feature_cols]

y=y_tr

x_tr=np.nan_to_num(x_tr)
x_te=np.nan_to_num(x_te)
#knn.fit(x.values, y.values)
knn.fit(x_tr, y)

pred_tr=knn.predict(x_tr)
pred_te=knn.predict(x_te)

# Find PassengerId:
l=list(x_te)
survived=list(pred_te)
f=open('wubin_submission.csv','w+')
f.write('PassengerId,Survived\n')
for i in range(len(l)):
    PassId=str(int(l[i][0]))
    f.write(PassId)
    f.write(',')
    f.write(str(survived[i]))
    f.write('\n')
f.close()

sc_tr=accuracy_score(y_tr,pred_tr)
print('accuracy_score of training set is: ', sc_tr)


### Logistic Regression
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
X=x_tr
logreg.fit(X, y)

# predict the response for new observations
#logreg.predict(X_new)

# pred. fit data
pred_tr_log=logreg.predict(X)

# accuracy
sc_tr_log=accuracy_score(y_tr,pred_tr_log)
print('accuracy_score of training set is: ', sc_tr_log)

