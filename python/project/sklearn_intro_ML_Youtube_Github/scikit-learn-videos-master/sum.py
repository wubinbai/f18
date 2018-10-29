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
# 
