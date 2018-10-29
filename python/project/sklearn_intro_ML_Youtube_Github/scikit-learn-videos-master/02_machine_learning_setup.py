
# coding: utf-8

# # Setting up Python for machine learning: scikit-learn and Jupyter Notebook ([video #2](https://www.youtube.com/watch?v=IsXXlYVBt1M&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=2))
# 
# Created by [Data School](http://www.dataschool.io/). Watch all 9 videos on [YouTube](https://www.youtube.com/playlist?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A). Download the notebooks from [GitHub](https://github.com/justmarkham/scikit-learn-videos).
# 
# **Note:** Since the video recording, the official name of the "IPython Notebook" was changed to "Jupyter Notebook". However, the functionality is the same.

# ## Agenda
# 
# - What are the benefits and drawbacks of scikit-learn?
# - How do I install scikit-learn?
# - How do I use the Jupyter Notebook?
# - What are some good resources for learning Python?

# ![scikit-learn algorithm map](images/02_sklearn_algorithms.png)

# ## Benefits and drawbacks of scikit-learn
# 
# ### Benefits:
# 
# - **Consistent interface** to machine learning models
# - Provides many **tuning parameters** but with **sensible defaults**
# - Exceptional **documentation**
# - Rich set of functionality for **companion tasks**
# - **Active community** for development and support
# 
# ### Potential drawbacks:
# 
# - Harder (than R) to **get started with machine learning**
# - Less emphasis (than R) on **model interpretability**
# 
# ### Further reading:
# 
# - Ben Lorica: [Six reasons why I recommend scikit-learn](http://radar.oreilly.com/2013/12/six-reasons-why-i-recommend-scikit-learn.html)
# - scikit-learn authors: [API design for machine learning software](http://arxiv.org/pdf/1309.0238v1.pdf)
# - Data School: [Should you teach Python or R for data science?](http://www.dataschool.io/python-or-r-for-data-science/)

# ![scikit-learn logo](images/02_sklearn_logo.png)

# ## Installing scikit-learn
# 
# **Option 1:** [Install scikit-learn library](http://scikit-learn.org/stable/install.html) and dependencies (NumPy and SciPy)
# 
# **Option 2:** [Install Anaconda distribution](https://www.anaconda.com/download/) of Python, which includes:
# 
# - Hundreds of useful packages (including scikit-learn)
# - IPython and Jupyter Notebook
# - conda package manager
# - Spyder IDE

# ![Jupyter logo](images/02_jupyter_logo.svg)

# ## Using the Jupyter Notebook
# 
# ### Components:
# 
# - **IPython interpreter:** enhanced version of the standard Python interpreter
# - **Browser-based notebook interface:** weave together code, formatted text, and plots
# 
# ### Installation:
# 
# - **Option 1:** [Install the Jupyter notebook](https://jupyter.readthedocs.io/en/latest/install.html) (includes IPython)
# - **Option 2:** Included with the Anaconda distribution
# 
# ### Launching the Notebook:
# 
# - Type **jupyter notebook** at the command line to open the dashboard
# - Don't close the command line window while the Notebook is running
# 
# ### Keyboard shortcuts:
# 
# **Command mode** (gray border)
# 
# - Create new cells above (**a**) or below (**b**) the current cell
# - Navigate using the **up arrow** and **down arrow**
# - Convert the cell type to Markdown (**m**) or code (**y**)
# - See keyboard shortcuts using **h**
# - Switch to Edit mode using **Enter**
# 
# **Edit mode** (green border)
# 
# - **Ctrl+Enter** to run a cell
# - Switch to Command mode using **Esc**
# 
# ### IPython, Jupyter, and Markdown resources:
# 
# - [nbviewer](http://nbviewer.jupyter.org/): view notebooks online as static documents
# - [IPython documentation](http://ipython.readthedocs.io/en/stable/)
# - [Jupyter Notebook quickstart](http://jupyter.readthedocs.io/en/latest/content-quickstart.html)
# - [GitHub's Mastering Markdown](https://guides.github.com/features/mastering-markdown/): short guide with lots of examples

# ## Resources for learning Python
# 
# - [Codecademy's Python course](https://www.codecademy.com/learn/learn-python): browser-based, tons of exercises
# - [DataQuest](https://www.dataquest.io/): browser-based, teaches Python in the context of data science
# - [Google's Python class](https://developers.google.com/edu/python/): slightly more advanced, includes videos and downloadable exercises (with solutions)
# - [Python for Everybody](https://www.py4e.com/): beginner-oriented book, includes slides and videos

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

