from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

from sklearn.datasets import load_iris

iris = load_iris()
X=iris.data
y=iris.target
X=X[:,:2]
y2 = y.reshape(y.shape[0],1)
z = np.concatenate([X,y2],axis=1)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 4)


from sklearn.metrics import accuracy_score

k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

want=input('Do you wannna print accuracy score for k=1 to 25 knn? Type "yes" or Enter to continue: ')
if want == 'yes':
    from pandas import Series
    s=Series(scores)
    print('accurarcy score is: ', s)

def plot_scatter(X,y):
    colors = ['r','g','b']
    for i in range(3):
        xs = X[:,0][y==i]
        ys = X[:,1][y==i]
        plt.scatter(xs,ys,c=colors[i])


