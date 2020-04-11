import os
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
import re

fileName='train.csv'
filePath = '..//data//titanic//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
df = pd.read_csv(csvPath, sep=',', engine='python')

# Get passenger total, total deaths, total survived
survivalData = df['Survived']
deaths=0
survived=0
total=len(survivalData)
for entry in survivalData:
    if entry == 0:
        deaths+=1
    elif entry == 1:
        survived+=1
    else:
        print(entry)

print('Survival')
print('Total: '+ str(total))
print('Deaths: ' + str(deaths))
print('Survived: ' + str(survived))
print()

#Passenger classes
classData = df['Pclass']
one=0
two=0
three=0
for entry in classData:
    if entry == 1:
        one+=1
    elif entry == 2:
        two+=1
    elif entry == 3:
        three+=1
    else:
        print(entry)

print('Passenger class')
print('First class: ' + str(one))
print('Second class: ' + str(two))
print('Third class: ' + str(three))
