import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

fileName='train_cleaned.csv'
filePath = '..//data//titanic_cleaned//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
df = pd.read_csv(csvPath, sep=',', engine='python')


y = df['Survived']
x = df[['Pclass', 'Sex' , 'AgeGroup', 'FamSize', 'isAlone']]#, 'CabinBool']]# 'Embarked']]#,'Parch', 'SibSp', 'TktNum']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, test_size=0.5, random_state=42)

x_test3=x_train2
y_test3=y_train2

x_train3=x_test.append(x_test2)
y_train3=y_test.append(y_test2)

x_train2=x_train2.append(x_test)
y_train2=y_train2.append(y_test)

#GAUSSIAN
model = GaussianNB().fit(x_train, y_train) #fitting our model
predicted_y = model.predict(x_test)
accuracy= accuracy_score(y_test, predicted_y)
print ('GAUSSIAN NAIVE BAYES: ' + str(accuracy))

# DECISION TREE
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
predicted_y=decisiontree.predict(x_test)
accuracy= accuracy_score(y_test, predicted_y)
print ('DECISION TREE: ' + str(accuracy))

#SVC
svc = SVC()
svc.fit(x_train, y_train)
predicted_y=svc.predict(x_test)
accuracy= accuracy_score(y_test, predicted_y)
print ('SVC: ' + str(accuracy))


# GRADIENT BOOSTING
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
predicted_y=gbk.predict(x_test)
accuracy= accuracy_score(y_test, predicted_y)
print ('GRADIENT BOOSTING: ' + str(accuracy))


# RANDOM FOREST
rf = RandomForestClassifier(criterion='gini',
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(x_train,y_train)
predicted_y=rf.predict(x_test)
accuracy= accuracy_score(y_test, predicted_y)
print ('RANDOM FOREST: ' + str(accuracy))



# TEST CLEANED
fileName='test_cleaned.csv'
filePath = '..//data//titanic_cleaned//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
df = pd.read_csv(csvPath, sep=',', engine='python')

'''
x =df[['Pclass', 'Sex' , 'AgeGroup', 'FamSize', 'isAlone']]
predicted_y=rf.predict(x)


# GET TEST UNCLEANED PassengerId INDEX
fileName='test.csv'
filePath = '..//data//titanic//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
test = pd.read_csv(csvPath, sep=',', engine='python')

d={'Survived': predicted_y}
prediction_df=pd.DataFrame(d, index=test['PassengerId'])
prediction_df.index.name='PassengerId'

# Save to csv
csvName='third_prediction.csv'
filePath = '..//data//predictions//' + csvName
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
prediction_df.to_csv(csvPath)



# GRADIENT BOOSTING
gbk = GradientBoostingClassifier()
gbk.fit(x_test2, y_test2)
predicted_y=gbk.predict(x_test2)
accuracy= accuracy_score(y_test2, predicted_y)
print ('GRADIENT BOOSTING: ' + str(accuracy))

# GRADIENT BOOSTING
gbk = GradientBoostingClassifier()
gbk.fit(x_test3, y_test3)
predicted_y=gbk.predict(x_test3)
accuracy= accuracy_score(y_test3, predicted_y)
print ('GRADIENT BOOSTING: ' + str(accuracy))


# TEST CLEANED
fileName='test_cleaned.csv'
filePath = '..//data//titanic_cleaned//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
df = pd.read_csv(csvPath, sep=',', engine='python')

x = df[['Pclass', 'Sex' , 'AgeGroup', 'Parch','SibSp', 'TktNum']]# 'Embarked']]#,'Parch', 'SibSp', 'TktNum']]
predicted_y=gbk.predict(x)


# GET TEST UNCLEANED PassengerId INDEX
fileName='test.csv'
filePath = '..//data//titanic//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
test = pd.read_csv(csvPath, sep=',', engine='python')

d={'Survived': predicted_y}
prediction_df=pd.DataFrame(d, index=test['PassengerId'])
prediction_df.index.name='PassengerId'

# Save to csv
csvName='first_prediction.csv'
filePath = '..//data//predictions//' + csvName
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
prediction_df.to_csv(csvPath)
'''
