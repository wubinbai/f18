import matplotlib
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

n=input('show image i: Please enter a number: ')
n=int(n)
some_digit = X[n]
para1 = int(some_digit.shape[0]**0.5)
some_digit_image = some_digit.reshape(para1,para1)
plt.imshow(some_digit_image,cmap = matplotlib.cm.binary, interpolation = "nearest")

plt.show()

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
sgd = SGDClassifier()
# Binary SGD:
sgd_5 = SGDClassifier()
y_train_5 = (y_train==5)
y_test_5 = (y_test==5)
sgd.fit(X_train,y_train)
sgd_5.fit(X_train,y_train_5)

predict = sgd.predict(X_test)
predict_5 = sgd_5.predict(X_test)
from sklearn.metrics import confusion_matrix
cf=confusion_matrix(y_test,predict)
cf_5=confusion_matrix(y_test_5,predict_5)
from sklearn.metrics import precision_score, recall_score, f1_score
ps5 = precision_score(y_test_5,predict_5)
rs5 = recall_score(y_test_5,predict_5)
f1s5 = f1_score(y_test_5,predict_5)
