# No need train data
# train = pd.read_csv('train.csv')
# use test PassengerId to predict that all ppl died, 0.
test = pd.read_csv('test.csv')


f = open('all_died_submission.csv','w+')
f.write('PassengerId,Survived\n')
for i in range(test.shape[0]):
    PassId=test.PassengerId[i]
    f.write(str(PassId))
    f.write(',')
    f.write(str(0))
    f.write('\n')
f.close()



