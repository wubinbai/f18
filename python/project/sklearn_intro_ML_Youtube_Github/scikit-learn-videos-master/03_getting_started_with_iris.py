
# coding: utf-8

# # Getting started in scikit-learn with the famous iris dataset ([video #3](https://www.youtube.com/watch?v=hd1W4CyPX58&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=3))
# 
# Created by [Data School](http://www.dataschool.io/). Watch all 9 videos on [YouTube](https://www.youtube.com/playlist?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A). Download the notebooks from [GitHub](https://github.com/justmarkham/scikit-learn-videos).
# 
# **Note:** This notebook uses Python 3.6 and scikit-learn 0.19.1. The original notebook (shown in the video) used Python 2.7 and scikit-learn 0.16, and can be downloaded from the [archive branch](https://github.com/justmarkham/scikit-learn-videos/tree/archive).

# ## Agenda
# 
# - What is the famous iris dataset, and how does it relate to machine learning?
# - How do we load the iris dataset into scikit-learn?
# - How do we describe a dataset using machine learning terminology?
# - What are scikit-learn's four key requirements for working with data?

# ## Introducing the iris dataset

# ![Iris](images/03_iris.png)

# - 50 samples of 3 different species of iris (150 samples total)
# - Measurements: sepal length, sepal width, petal length, petal width

# In[4]:


from IPython.display import IFrame
print(type(IFrame))
IFrame('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', width=300, height=200)


# ## Machine learning on the iris dataset
# 
# - Framed as a **supervised learning** problem: Predict the species of an iris using the measurements
# - Famous dataset for machine learning because prediction is **easy**
# - Learn more about the iris dataset: [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Iris)

# ## Loading the iris dataset into scikit-learn

# In[5]:


# import load_iris function from datasets module
from sklearn.datasets import load_iris


# In[6]:


# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)


# In[14]:


# print the iris data
print(len(iris.data))
print(iris.data.shape)
print(type(iris.data))
print(iris.data)



# ## Machine learning terminology
# 
# - Each row is an **observation** (also known as: sample, example, instance, record)
# - Each column is a **feature** (also known as: predictor, attribute, independent variable, input, regressor, covariate)

# In[20]:


# print the names of the four features
#print(iris.feature_names,iris.DESCR)
dir(iris)


# In[7]:


# print integers representing the species of each observation
print(iris.target)


# In[8]:


# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)


# - Each value we are predicting is the **response** (also known as: target, outcome, label, dependent variable)
# - **Classification** is supervised learning in which the response is categorical
# - **Regression** is supervised learning in which the response is ordered and continuous

# ## Requirements for working with data in scikit-learn
# 
# 1. Features and response are **separate objects**
# 2. Features and response should be **numeric**
# 3. Features and response should be **NumPy arrays**
# 4. Features and response should have **specific shapes**

# In[21]:


# check the types of the features and response
print(type(iris.data))
print(type(iris.target))


# In[22]:


# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print(iris.data.shape)


# In[11]:


# check the shape of the response (single dimension matching the number of observations)
print(iris.target.shape)


# In[23]:


# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target


# ## Resources
# 
# - scikit-learn documentation: [Dataset loading utilities](http://scikit-learn.org/stable/datasets/)
# - Jake VanderPlas: Fast Numerical Computing with NumPy ([slides](https://speakerdeck.com/jakevdp/losing-your-loops-fast-numerical-computing-with-numpy-pycon-2015), [video](https://www.youtube.com/watch?v=EEUXKG97YRw))
# - Scott Shell: [An Introduction to NumPy](http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf) (PDF)

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

