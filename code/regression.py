#import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

fileName='dataset_for_regression.csv'
filePath = '..//data//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
data= pd.read_csv(csvPath, engine='python')
print(data.iloc[:,2])
data.iloc[:,2] = data.iloc[:,2].replace({':':'.'}, regex=True)
data.iloc[:,2] = pd.to_numeric(data.iloc[:,2], downcast="float")
data.iloc[:,2] =data.iloc[:,2].round(0)
print(data.iloc[:,2])

data.head()

X = data.iloc[:,0:-1] # X is the features in our dataset
y = data.iloc[:,-1]   # y is the Labels in our dataset

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = GaussianNB().fit(X_train, y_train) #fitting our model
predicted_y = model.predict(X_test)
accuracy_score = accuracy_score(y_test, predicted_y)
print (accuracy_score)
