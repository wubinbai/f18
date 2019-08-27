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


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

feats = pd.concat([X.Sex,X.Pclass,X.SibSp,X.Parch,X.Embarked],axis=1)
dt.fit(feats,y)
test_feats = pd.concat([test.Sex,test.Pclass,test.SibSp,test.Parch,test.Embarked],axis=1)
y_train_pred = dt.predict(feats)
df_train_pred = pd.concat([feats,pd.DataFrame(y_train_pred)],axis=1) 
y_pred = dt.predict(test_feats)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.set_index(test.PassengerId,inplace=True)
y_pred_df.columns = ['Survived']
y_pred_df.to_csv("dt_embarked_sex_pclass_sibsp_parch.csv")

from sklearn.tree import export_graphviz

import os
PROJECT_ROOT_DIR= '.'
CHAPTER_ID = 'decision_trees'

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)


export_graphviz(
        dt,
        out_file=image_path("dt_embarked_sex_pclass_sibsp_parch.dot"),
       # feature_names = ['Sex','Pclass'], #iris.feature_names[:], #[2:] to [:]
        class_names = ['Died','Survived'],#iris.target_names,
        rounded=True,
        filled=True)

