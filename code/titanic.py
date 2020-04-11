import os
import pandas as pd
import matplotlib.pyplot as plt
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

# Servival per passenger class
firstDeath=0
firstSurvived=0
secondDeath=0
secondSurvived=0
thirdDeath=0
thirdSurvived=0
for i, entry in enumerate(classData):
    if entry == 1:
        if survivalData.loc[i] == 0:
            firstDeath+=1
        elif survivalData.loc[i] ==1:
            firstSurvived+=1
    elif entry == 2:
        if survivalData.loc[i] == 0:
            secondDeath+=1
        elif survivalData.loc[i] ==1:
            secondSurvived+=1
    elif entry == 3:
        if survivalData.loc[i] == 0:
            thirdDeath+=1
        elif survivalData.loc[i] ==1:
            thirdSurvived+=1
deathTotals = (firstDeath, secondDeath, thirdDeath)
survivalTotals = (firstSurvived, secondSurvived, thirdSurvived)

# Barchart that shows #deaths & #survival per passenger class
N=3
fig, ax = plt.subplots()
ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, survivalTotals, width, bottom=0)
p2 =  ax.bar(ind+width, deathTotals, width, bottom=0)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('First', 'Second', 'Third'))
ax.legend((p1[0], p2[0]), ('Death', 'Survived'))

# Save plot
filePath='..//plots//titanic//'
fileDir = os.path.dirname(os.path.realpath('__file__'))
plotPath = os.path.join(fileDir, filePath) #'../data/test.csv')
plotPath = os.path.abspath(os.path.realpath(plotPath))
plotName = 'ClassSurvivalBarChart.png'
plt.savefig(os.path.join(plotPath, plotName), bbox_inches='tight')
plt.show()
