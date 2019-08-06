from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X,y)

from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self,X,y=None):
        pass
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

never_5_clf = Never5Classifier()
score_never_5_clf = cross_val_score(never_5_clf,X_train,y_train_5,cv=3,scoring="accuracy")

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(never_5_clf,X_train,y_train_5,cv=3)

from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y_train_5,y_train_pred)


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=0)

y_train_pred_sgd = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)

cf_sgd = confusion_matrix(y_train_5, y_train_pred_sgd)
