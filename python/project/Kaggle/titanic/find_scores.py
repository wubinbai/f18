# if we want precision, recall and confusion_matrix, or f1_score, def fun.
from sklearn.metrics import recall_score, precision_score, f1_score
def score_list(y_pred, y_true):

    precision = precision_score(y_pred,y_true)
    recall = recall_score(y_pred,y_true)
    f1 = f1_score(y_pred,y_true)
    print("Precision is: ", precision)
    print("Recall is: ", recall)
    print("f1 score is: ", f1)

train = pd.read_csv("train.csv")
y = train[train.columns[1]]

'''
all_died = pd.read_csv("all_died_submission.csv")
all_died.set_index(all_died.PassengerId)
all_died.drop(['PassengerId'],axis=1,inplace=True)
'''
# predict for train data
all_zero = pd.Series(np.zeros(y.shape[0]))
# compare to train data
tf_all_died = (y == all_zero)
# accuracy for train data predict
score_all_died = tf_all_died.sum() / tf_all_died.shape[0]

print('scores_all_died: ', score_all_died)
print("analytics of all_died: ")
#score_list(y,all_zero)

# read male died train csv
male_died = pd.read_csv("male_died_train.csv")
# compare to train target
score_male_died = (male_died.Survived == y).sum() / y.shape[0]

print('scores_male_died: ', score_male_died)
print("analytics of male_died: ")
score_list(y,male_died.Survived)
