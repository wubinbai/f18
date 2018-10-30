# sum.py
####################
# sec 1
# What is ML?
####################
# sec 2
# Pros & Cons of Sklearn; 2 components of IPython Notebook: a) Kernel b) Browser; nbviewer.ipython.org is discussed
####################
# sec 3
# Equiv. saying:
# row: obs = sample = e.g. = instance = record
# column: feature = predictor = indep. var. = input = regressor = covariate
# val. we pred. = target = response = outcome = label = dep. var
####################
# sec 4
# knn: choose k, find k nearest, predict
# Paradigm: 4 steps: import, instantiate, train(fit), predict
# sec 5
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
knn = KNeighborsClassifier(n_neighbors=1)
logreg = LogisticRegression()
####################
# sec 5
# metrics
from sklearn.metrics import accuracy_score
# to use it:
# accuracy_score(y_true,y_pred)
# problem: e.g. train and test on the same data with knn has problem of overfitting, since it fits very well on known data, too complex model that cannot generalize to out-of-sample data.
# resolution: split data(train/test split)
# 3 Steps: a) split b)train on training se c) test on testing set
# import train/test split from sklearn
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4)
# train(fit) with training set...
# test(accuracy_score) with testing set...

