train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")
t=train
y=train.Survived
X = train.copy()
X.drop(columns='Survived',axis=1,inplace=True)
X=X.fillna(-999999)
X_copy = X.copy()
for c in train.columns[train.dtypes=='object']:
    X[c]=X[c].factorize()[0]
