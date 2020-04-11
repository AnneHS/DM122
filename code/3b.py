import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#READ FILE ABOUT LIFE EXPECTANCY 
fileName='Life Expectancy Data2.csv'
filePath = '..//data//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
df = pd.read_csv(csvPath, sep=';', engine='python')

print(df.head())
df.fillna(df.mean(), inplace=True)

X = df.iloc[:,1:] # X is the features in our dataset
y = df.iloc[:,0]   # y is the Labels in our dataset

#set up 3-cross validation for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

X_test3 = X_train2
y_test3 = y_train2

X_train3 = X_test.append(X_test2)
y_train3 = y_test.append(y_test2)
print(len(X_train), len( X_train2), len( X_train3), len(X))
print(len(y_train), len(y_train2), len(y_train3), len(y))

X_train2 = X_train2.append(X_test) 
y_train2 = y_train2.append(y_test)

#LINEAR REGRESSION SCORES FOR 3-FOLD CROSS VALIDATIONN

ls = LinearRegression()
ls.fit(X_train, y_train)
print(ls.score(X_test, y_test))
ls.fit(X_train2, y_train2)
print(ls.score(X_test2, y_test2))
ls.fit(X_train3, y_train3)
print(ls.score(X_test3, y_test3))
predicted_y = ls.predict(X_test)

# The coefficients for each x attribute
print('Coefficients: \n', ls.coef_)

# The mean SQUARED error
print('Mean SQUARED error: %.2f'% mean_squared_error(y_test, predicted_y))
# The mean ABSOLUTE error
print('Mean ABSOLUTE error: %.2f'% mean_absolute_error(y_test, predicted_y))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'% r2_score(y_test, predicted_y))


plt.scatter(predicted_y, y_test)
plt.xlabel("predicted")
plt.ylabel("observed")
plt.show()


plt.scatter(np.arange(len(y_test)), abs(predicted_y - y_test))
plt.ylabel("Samples", fontsize = 13)
plt.xlabel("Absolute residual", fontsize=13)
plt.show()

#TRY RANSAC FIT...IGNORES OUTLIERS A BIT MORE

ransac = RANSACRegressor()
ransac.fit(X_train, y_train)
print(ransac.score(X_test, y_test))

# Predict data of estimated models
predicted_y = ransac.predict(X_test)

# Compare estimated coefficients
print("Estimated coefficients (true, linear regression, RANSAC):")
print(ransac.estimator_.coef_)

print('Mean SQUARE error: %.2f'% mean_squared_error(y_test, predicted_y))
print('Mean ABSOLUTE error: %.2f'% mean_absolute_error(y_test, predicted_y))
print('Coefficient of determination: %.2f'% r2_score(y_test, predicted_y))


#TRY OTHER SIMPLE LINEAR REGRESSOR
from sklearn.linear_model import ElasticNet

model = ElasticNet().fit(X_train, y_train)
model.score(X_test, y_test)


predicted_y = gpr.predict(X_test)
print("Estimated coefficients gpr:")
#print(gpr.estimator_.coef_)

print('Mean SQUARE error: %.2f'% mean_squared_error(y_test, predicted_y))
print('Mean ABSOLUTE error: %.2f'% mean_absolute_error(y_test, predicted_y))
print('Coefficient of determination: %.2f'% r2_score(y_test, predicted_y))


# TRY OTHER REGRESSOR THAT IGNORES OUTLIERS
from sklearn.linear_model import TheilSenRegressor

model = TheilSenRegressor().fit(X_train, y_train)
model.score(X_test, y_test)


predicted_y = model.predict(X_test)

print('Mean SQUARE error: %.2f'% mean_squared_error(y_test, predicted_y))
print('Mean ABSOLUTE error: %.2f'% mean_absolute_error(y_test, predicted_y))
print('Coefficient of determination: %.2f'% r2_score(y_test, predicted_y))