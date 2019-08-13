from sklearn.preprocessing import LabelBinarizer

train = pd.read_csv('train.csv')
# use test PassengerId to predict that all ppl died, 0.
test = pd.read_csv('test.csv')
test0 = test.copy()

lb = LabelBinarizer()
test.Sex = lb.fit_transform(test.Sex)
train.Sex = lb.fit_transform(train.Sex)

f = open('male_died_submission.csv','w+')
f.write('PassengerId,Survived\n')
for i in range(test.shape[0]):
    PassId=test.PassengerId[i]
    f.write(str(PassId))
    f.write(',')
    if test.Sex[i] == 0:
        f.write(str(1))
    else:
        f.write(str(0))
    f.write('\n')
f.close()

# write predictions for train:
f = open('male_died_train.csv','w+')
f.write('PassengerId,Survived\n')
for i in range(train.shape[0]):
    PassId=train.PassengerId[i]
    f.write(str(PassId))
    f.write(',')
    if train.Sex[i] == 0:
        f.write(str(1))
    else:
        f.write(str(0))
    f.write('\n')
f.close()


