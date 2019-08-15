train = pd.read_csv("train.csv")
t=train
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

import seaborn as sns
def plot_simple(train):
    sns.countplot(x='Survived',data=train)
    sns.countplot(x='Pclass',data=train)
    sns.countplot(x='Embarked',data=train)
def correlation_plot(train):
    sns.catplot(x='Sex',col='Survived',kind='count',data=train)
    sns.catplot('Pclass','Survived',kind='point',data=train)
    sns.catplot('Pclass','Survived',hue='Sex',kind='point',data=train)
    sns.catplot(x='Survived',col='Embarked',kind='count',data=train)
# don't know why this will have wrong plot    sns.catplot(x='Sex',col='Embarked',kind='count',data=train)
    sns.catplot(x='Pclass',col='Embarked',plot='count',data=train)
