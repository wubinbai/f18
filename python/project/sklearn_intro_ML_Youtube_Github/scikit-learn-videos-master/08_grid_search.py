
# coding: utf-8

# # Efficiently searching for optimal tuning parameters ([video #8](https://www.youtube.com/watch?v=Gol_qOgRqfA&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=8))
# 
# Created by [Data School](http://www.dataschool.io/). Watch all 9 videos on [YouTube](https://www.youtube.com/playlist?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A). Download the notebooks from [GitHub](https://github.com/justmarkham/scikit-learn-videos).
# 
# **Note:** This notebook uses Python 3.6 and scikit-learn 0.19.1. The original notebook (shown in the video) used Python 2.7 and scikit-learn 0.16, and can be downloaded from the [archive branch](https://github.com/justmarkham/scikit-learn-videos/tree/archive).

# ## Agenda
# 
# - How can K-fold cross-validation be used to search for an **optimal tuning parameter**?
# - How can this process be made **more efficient**?
# - How do you search for **multiple tuning parameters** at once?
# - What do you do with those tuning parameters before making **real predictions**?
# - How can the **computational expense** of this process be reduced?

# ## Review of K-fold cross-validation

# Steps for cross-validation:
# 
# - Dataset is split into K "folds" of **equal size**
# - Each fold acts as the **testing set** 1 time, and acts as the **training set** K-1 times
# - **Average testing performance** is used as the estimate of out-of-sample performance
# 
# Benefits of cross-validation:
# 
# - More **reliable** estimate of out-of-sample performance than train/test split
# - Can be used for selecting **tuning parameters**, choosing between **models**, and selecting **features**
# 
# Drawbacks of cross-validation:
# 
# - Can be computationally **expensive**

# ## Review of parameter tuning using `cross_val_score`

# **Goal:** Select the best tuning parameters (aka "hyperparameters") for KNN on the iris dataset

# In[2]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target


# In[4]:


# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)


# In[5]:


# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[6]:


# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)


# In[7]:


# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# ## More efficient parameter tuning using `GridSearchCV`

# Allows you to define a **grid of parameters** that will be **searched** using K-fold cross-validation

# In[8]:


from sklearn.model_selection import GridSearchCV


# In[9]:


# define the parameter values that should be searched
k_range = list(range(1, 31))
print(k_range)


# In[10]:


# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)
print(param_grid)


# In[11]:


# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)


# - You can set **`n_jobs = -1`** to run computations in parallel (if supported by your computer and OS)

# In[12]:


# fit the grid with data
grid.fit(X, y)


# In[13]:


# view the results as a pandas DataFrame
import pandas as pd
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]


# In[14]:


# examine the first result
print(grid.cv_results_['params'][0])
print(grid.cv_results_['mean_test_score'][0])


# In[15]:


# print the array of mean scores only
grid_mean_scores = grid.cv_results_['mean_test_score']
print(grid_mean_scores)


# In[16]:


# plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# In[17]:


# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


# ## Searching multiple parameters simultaneously

# - **Example:** tuning `max_depth` and `min_samples_leaf` for a `DecisionTreeClassifier`
# - Could tune parameters **independently**: change `max_depth` while leaving `min_samples_leaf` at its default value, and vice versa
# - But, best performance might be achieved when **neither parameter** is at its default value

# In[18]:


# define the parameter values that should be searched
k_range = list(range(1, 31))
weight_options = ['uniform', 'distance']


# In[19]:


# create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)


# In[20]:


# instantiate and fit the grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False)
grid.fit(X, y)


# In[21]:


# view the results
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]


# In[22]:


# examine the best model
print(grid.best_score_)
print(grid.best_params_)


# ## Using the best parameters to make predictions

# In[23]:


# train your model using all data and the best known parameters
knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
knn.fit(X, y)

# make a prediction on out-of-sample data
knn.predict([[3, 5, 4, 2]])


# In[24]:


# shortcut: GridSearchCV automatically refits the best model using all of the data
grid.predict([[3, 5, 4, 2]])


# ## Reducing computational expense using `RandomizedSearchCV`

# - Searching many different parameters at once may be computationally infeasible
# - `RandomizedSearchCV` searches a subset of the parameters, and you control the computational "budget"

# In[25]:


from sklearn.model_selection import RandomizedSearchCV


# In[26]:


# specify "parameter distributions" rather than a "parameter grid"
param_dist = dict(n_neighbors=k_range, weights=weight_options)


# - **Important:** Specify a continuous distribution (rather than a list of values) for any continous parameters

# In[27]:


# n_iter controls the number of searches
rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5, return_train_score=False)
rand.fit(X, y)
pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]


# In[28]:


# examine the best model
print(rand.best_score_)
print(rand.best_params_)


# In[29]:


# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
best_scores = []
for _ in range(20):
    rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, return_train_score=False)
    rand.fit(X, y)
    best_scores.append(round(rand.best_score_, 3))
print(best_scores)


# ## Resources
# 
# - scikit-learn documentation: [Grid search](http://scikit-learn.org/stable/modules/grid_search.html), [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html), [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
# - Timed example: [Comparing randomized search and grid search](http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html)
# - scikit-learn workshop by Andreas Mueller: [Video segment on randomized search](https://youtu.be/0wUF_Ov8b0A?t=17m38s) (3 minutes), [related notebook](https://github.com/amueller/pydata-nyc-advanced-sklearn/blob/master/Chapter%203%20-%20Randomized%20Hyper%20Parameter%20Search.ipynb)
# - Paper by Yoshua Bengio: [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)

# ## Comments or Questions?
# 
# - Email: <kevin@dataschool.io>
# - Website: http://dataschool.io
# - Twitter: [@justmarkham](https://twitter.com/justmarkham)

# In[1]:


from IPython.core.display import HTML
def css_styling():
    styles = open("styles/custom.css", "r").read()
    return HTML(styles)
css_styling()

