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
for c in test.columns[test.dtypes=='object']:
    test[c]=test[c].factorize()[0]


from sklearn.ensemble import RandomForestClassifier as RF
rf = RF()
sex_pclass = pd.concat([X.Sex,X.Pclass],axis=1)
rf.fit(sex_pclass,y)
test_sex_pclass = pd.concat([test.Sex,test.Pclass],axis=1)
y_pred = rf.predict(test_sex_pclass)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.set_index(test.PassengerId,inplace=True)
y_pred_df.columns = ['Survived']
y_pred_df.to_csv("rf_sex_pclass.csv")


