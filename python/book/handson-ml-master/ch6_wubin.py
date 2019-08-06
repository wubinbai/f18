from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:,:] #[:,2:] originally
y = iris.target

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X,y)

from sklearn.tree import export_graphviz

import os
PROJECT_ROOT_DIR= '.'
CHAPTER_ID = 'decision_trees'

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)


export_graphviz(
        tree_clf,
        out_file=image_path("iris_all.dot"),
        feature_names=iris.feature_names[:], #[2:] to [:]
        class_names=iris.target_names,
        rounded=True,
        filled=True       
                
        )
