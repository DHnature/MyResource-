import pandas 
titanic=pandas.read_csv('E:/Coding/python/workplace/titanic.csv')
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
titanic.loc[titanic["Sex"]=="male","Sex"]=0
titanic.loc[titanic["Sex"]=="female","Sex"]=1
titanic["Embarked"]=titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2
titanic.head(100)
print(titanic.describe())
titanic["Sex"].unique()


from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
alg=LinearRegression()
kf=KFold(titanic.shape[0],n_folds=3,random_state=1)
predictions=[

for train, test in kf:
    train_predictors=(titanic[predictors].iloc[train,:])
    train_target=titanic["Survived"].iloc[train]
    alg.fit(train_predictors,train_target)
    test_predictions=alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
import numpy as np
predictions=np.concatenate(predictions,axis=0)
predictions_new = list(range(len(predictions)))
titanic_survived_list = list(titanic['Survived'])
summ = 0
for i in list(range(len(predictions))):

    if predictions[i] > 0.5:
        predictions_new[i] = 1
    else:
        predictions_new[i] = 0
    if predictions_new[i] == titanic_survived_list[i]:
        summ = summ+1
print (summ)
# print predictions_new

# print titanic['Survived']
accuracy = summ/float(len(predictions))
print (accuracy)
