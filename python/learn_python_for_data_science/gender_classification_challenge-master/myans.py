from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# 1
clf1 =  SVC(kernel="linear", C=0.025)  
# 2
clf2 = GaussianNB()
# 3
clf3 = KNeighborsClassifier(3)

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

prediction = clf.predict([[190, 70, 43]])
prediction1 = clf1.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])
prediction3 = clf3.predict([[190, 70, 43]])

# CHALLENGE compare their reusults and print the best one!
X_test = X #[[190, 70, 43]]
y_test = Y
#y_test = prediction
#y_test1 = prediction1
#y_test2 = prediction2
#y_test3 = prediction3

score = clf.score(X_test, y_test)
score1 = clf1.score(X_test, y_test)
score2 = clf2.score(X_test, y_test)
score3 = clf3.score(X_test, y_test)
print(prediction)
print('score is: ',score)
print(prediction1)
print('score is: ',score1)
print(prediction2)
print('score is: ',score2)
print(prediction3)
print('score is: ',score3)
