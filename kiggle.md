import pandas as pa
import numpy as np
titanic=pa.read_excel("E:/titanic2.xls")
titanic["age"]=titanic["age"].fillna(titanic["age"].median())
titanic["sex"].unique()
titanic.loc[titanic["sex"]=="male","sex"]=0
titanic.loc[titanic["sex"]=="female","sex"]=1
titanic["embarked"]=titanic["embarked"].fillna("S")
titanic.loc[titanic["embarked"]=="S","embarked"]=0
titanic.loc[titanic["embarked"]=="Q","embarked"]=1
titanic.loc[titanic["embarked"]=="C","embarked"]=2
titanic["fare"]=titanic["fare"].fillna(titanic["fare"].median())

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
predictors=["pclass","sex","age","sibsp","parch","fare","embarked"]
print(titanic.describe())

#检出输入数据的正确性


alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train, :])
    train_target = titanic["survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)



predictions = np.concatenate(predictions, axis=0)
predictions_new = list(range(len(predictions)))
titanic_survived_list = list(titanic['survived'])
summ = 0
for i in list(range(len(predictions))):

    if predictions[i] > 0.5:
        predictions_new[i] = 1
    else:
        predictions_new[i] = 0
    if predictions_new[i] == titanic_survived_list[i]:
        summ = summ + 1
print(summ)
# print predictions_new

# print titanic['Survived']
accuracy = summ / float(len(predictions))
print(accuracy)
