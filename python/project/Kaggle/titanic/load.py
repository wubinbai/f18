train = pd.read_csv("train.csv")
X = train.copy()
X.drop(columns='Survived',axis=1,inplace=True)
X=X.fillna(-999999)
X_copy = X.copy()
for c in train.columns[train.dtypes=='object']:
    X[c]=X[c].factorize()[0]
# You get error when using LabelEncoder for diff. types:

#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#for c in train.columns[train.dtypes=='object']:
#    le.fit_transform(X_copy[c])
