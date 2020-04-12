import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import math

fileName='train.csv'
filePath = '..//data//titanic//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
df = pd.read_csv(csvPath, sep=',', engine='python')

'''
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
ax.set_xticklabels(('Upper', 'Middle', 'Lower'))
ax.legend((p1[0], p2[0]), ('Survived', 'Death'))

# Save plot
filePath='..//plots//titanic//'
fileDir = os.path.dirname(os.path.realpath('__file__'))
plotPath = os.path.join(fileDir, filePath) #'../data/test.csv')
plotPath = os.path.abspath(os.path.realpath(plotPath))
plotName = 'ClassSurvivalBarChart.png'
plt.savefig(os.path.join(plotPath, plotName), bbox_inches='tight')
plt.show()

genderData=df['Sex']
maleDeath=0
maleSurvived=0
femaleDeath=0
femaleSurvived=0
for i, entry in enumerate(genderData):
    if entry == 'male':
        if survivalData.loc[i] == 0:
            maleDeath+=1
        elif survivalData.loc[i] ==1:
            maleSurvived+=1
    elif entry == 'female':
        if survivalData.loc[i] == 0:
            femaleDeath+=1
        elif survivalData.loc[i] ==1:
            femaleSurvived+=1

deathTotals = (maleDeath, femaleDeath)
survivalTotals = (maleSurvived, femaleSurvived)

# Barchart that shows #deaths & #survival per passenger class
N=2
fig, ax = plt.subplots()
ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, survivalTotals, width, bottom=0)
p2 =  ax.bar(ind+width, deathTotals, width, bottom=0)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Male', 'Female'))
ax.legend((p1[0], p2[0]), ('Survived', 'Death'))

# Save plot
filePath='..//plots//titanic//'
fileDir = os.path.dirname(os.path.realpath('__file__'))
plotPath = os.path.join(fileDir, filePath) #'../data/test.csv')
plotPath = os.path.abspath(os.path.realpath(plotPath))
plotName = 'GenderSurvivalBarChart.png'
plt.savefig(os.path.join(plotPath, plotName), bbox_inches='tight')
plt.show()

'''

ageData = df['Age']
x=[]
y=[]
for i, entry in enumerate(ageData):
    if isinstance(entry, int):
        x.app

unknown=0
cabinData = df['Cabin']

loc = np.where(pd.isnull(cabinData))
rows=loc[0]

deck_dict={'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7}
decks=[]
numbers=[]
for i, entry in enumerate(cabinData):
    if i in rows:
        decks.append(-1)
        numbers.append(-1)
    else:
        cabins = entry.split()
        if len(cabins)==1:
            d=deck_dict[cabins[0][0]]
            decks.append(d)
            numbers.append(cabins[0][1:])
        else:
            ds=[]
            ns=[]
            for cabin in cabins:
                if cabin[1:]=='':
                    n=-1
                else:
                    n=int(cabin[1:])
                ds.append(deck_dict[cabin[0]])
                ns.append(n)
            decks.append(ds)
            numbers.append(ns)

d={'Decks':decks, 'Numbers': numbers}
cabins_df = pd.DataFrame(d)

csvName='cabins_cleaned.csv'
filePath = '..//data//titanic_cleaned//' + csvName
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
cabins_df.to_csv(csvPath)
