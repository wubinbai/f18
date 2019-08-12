
# coding: utf-8

# # Training a machine learning model with scikit-learn ([video #4](https://www.youtube.com/watch?v=RlQuVL6-qe8&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=4))
# 
# Created by [Data School](http://www.dataschool.io/). Watch all 9 videos on [YouTube](https://www.youtube.com/playlist?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A). Download the notebooks from [GitHub](https://github.com/justmarkham/scikit-learn-videos).
# 
# **Note:** This notebook uses Python 3.6 and scikit-learn 0.19.1. The original notebook (shown in the video) used Python 2.7 and scikit-learn 0.16, and can be downloaded from the [archive branch](https://github.com/justmarkham/scikit-learn-videos/tree/archive).

# ## Agenda
# 
# - What is the **K-nearest neighbors** classification model?
# - What are the four steps for **model training and prediction** in scikit-learn?
# - How can I apply this pattern to **other machine learning models**?

# ## Reviewing the iris dataset

# In[2]:


from IPython.display import IFrame
IFrame('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', width=300, height=200)


# - 150 **observations**
# - 4 **features** (sepal length, sepal width, petal length, petal width)
# - **Response** variable is the iris species
# - **Classification** problem since response is categorical
# - More information in the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Iris)

# ## K-nearest neighbors (KNN) classification

# 1. Pick a value for K.
# 2. Search for the K observations in the training data that are "nearest" to the measurements of the unknown iris.
# 3. Use the most popular response value from the K nearest neighbors as the predicted response value for the unknown iris.

# ### Example training data
# 
# ![Training data](images/04_knn_dataset.png)

# ### KNN classification map (K=1)
# 
# ![1NN classification map](images/04_1nn_map.png)

# ### KNN classification map (K=5)
# 
# ![5NN classification map](images/04_5nn_map.png)

# *Image Credits: [Data3classes](http://commons.wikimedia.org/wiki/File:Data3classes.png#/media/File:Data3classes.png), [Map1NN](http://commons.wikimedia.org/wiki/File:Map1NN.png#/media/File:Map1NN.png), [Map5NN](http://commons.wikimedia.org/wiki/File:Map5NN.png#/media/File:Map5NN.png) by Agor153. Licensed under CC BY-SA 3.0*

# ## Loading the data

# In[6]:


# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()

# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target


# In[7]:


# print the shapes of X and y
print(X.shape)
print(y.shape)


# ## scikit-learn 4-step modeling pattern

# **Step 1:** Import the class you plan to use

# In[2]:


from sklearn.neighbors import KNeighborsClassifier


# **Step 2:** "Instantiate" the "estimator"
# 
# - "Estimator" is scikit-learn's term for model
# - "Instantiate" means "make an instance of"

# In[3]:


knn = KNeighborsClassifier(n_neighbors=1)
type(knn)


# - Name of the object does not matter
# - Can specify tuning parameters (aka "hyperparameters") during this step
# - All parameters not specified are set to their defaults

# In[4]:


print(knn)


# **Step 3:** Fit the model with data (aka "model training")
# 
# - Model is learning the relationship between X and y
# - Occurs in-place

# In[8]:


knn.fit(X, y)


# **Step 4:** Predict the response for a new observation
# 
# - New observations are called "out-of-sample" data
# - Uses the information it learned during the model training process

# In[10]:


knn.predict([[3, 5, 4, 2]])


# - Returns a NumPy array
# - Can predict for multiple observations at once

# In[11]:


X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
knn.predict(X_new)


# ## Using a different value for K

# In[12]:


# instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X, y)

# predict the response for new observations
knn.predict(X_new)


# ## Using a different classification model

# In[13]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response for new observations
logreg.predict(X_new)


# ## Resources
# 
# - [Nearest Neighbors](http://scikit-learn.org/stable/modules/neighbors.html) (user guide), [KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) (class documentation)
# - [Logistic Regression](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) (user guide), [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (class documentation)
# - [Videos from An Introduction to Statistical Learning](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/)
#     - Classification Problems and K-Nearest Neighbors (Chapter 2)
#     - Introduction to Classification (Chapter 4)
#     - Logistic Regression and Maximum Likelihood (Chapter 4)

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

