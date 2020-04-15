import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import math

# TRAIN DATA
fileName='train.csv'
filePath = '..//data//titanic//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
df = pd.read_csv(csvPath, sep=',', engine='python')

###############################################################################
# DESCRIBE CSV
###############################################################################
described = df.describe()
csvName='train_described.csv'
filePath = '..//data//titanic//' + csvName
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
described.to_csv(csvPath)

###############################################################################
# CORRELATION CSV
###############################################################################
correlation = df.corr()
csvName='train_correlation.csv'
filePath = '..//data//titanic//' + csvName
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
correlation.to_csv(csvPath)


###############################################################################
# GENERAL INFO
###############################################################################
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


###############################################################################
# PASSENGER CLASS vs SURVIVAL
###############################################################################
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
#plt.show()


###############################################################################
# GENDER vs SURVIVAL
###############################################################################
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
#plt.show()

###############################################################################
# EMBARKED VS SURVIVAL
###############################################################################

embarkedData = df['Embarked']
cSurvived=0
cDeath=0
qSurvived=0
qDeath=0
sSurvived=0
sDeath=0
for i, entry in enumerate(embarkedData):
    if entry == 'C':
        if survivalData.loc[i] == 0:
            cDeath+=1
        elif survivalData.loc[i] ==1:
            cSurvived+=1
    elif entry == 'Q':
        if survivalData.loc[i] == 0:
            qDeath+=1
        elif survivalData.loc[i] ==1:
            qSurvived+=1
    elif entry == 'S':
        if survivalData.loc[i] == 0:
            sDeath+=1
        elif survivalData.loc[i] ==1:
            sSurvived+=1

deathTotals = (cDeath, qDeath, sDeath)
survivalTotals = (cSurvived, qSurvived, sSurvived)

N=3
fig, ax = plt.subplots()
ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, survivalTotals, width, bottom=0)
p2 =  ax.bar(ind+width, deathTotals, width, bottom=0)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('C', 'Q', 'S'))
ax.legend((p1[0], p2[0]), ('Survived', 'Death'))

# Save plot
filePath='..//plots//titanic//'
fileDir = os.path.dirname(os.path.realpath('__file__'))
plotPath = os.path.join(fileDir, filePath) #'../data/test.csv')
plotPath = os.path.abspath(os.path.realpath(plotPath))
plotName = 'EmbarkedSurvivalBarChart.png'
plt.savefig(os.path.join(plotPath, plotName), bbox_inches='tight')
#plt.show()


###############################################################################
###############################################################################
# CLEANED DATASET
###############################################################################
###############################################################################

# TRAIN DATA
fileName='train_cleaned.csv'
filePath = '..//data//titanic_cleaned//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
df = pd.read_csv(csvPath, sep=',', engine='python')

###############################################################################
# AGEGROUPED
###############################################################################
ageData=df['AgeGroup']
print(ageData)
cDeath=0
cSurvived=0
tDeath=0
tSurvived=0
yaDeath=0
yaSurvived=0
aDeath=0
aSurvived=0
sDeath=0
sSurvived=0
uDeath=0
uSurvived=0

for i, entry in enumerate(ageData):
    if entry == 0:
        if survivalData.loc[i] == 0:
            cDeath+=1
        elif survivalData.loc[i] ==1:
            cSurvived+=1
    elif entry == 1:
        if survivalData.loc[i] == 0:
            tDeath+=1
        elif survivalData.loc[i] ==1:
            tSurvived+=1
    elif entry == 2:
        if survivalData.loc[i] == 0:
            yaDeath+=1
        elif survivalData.loc[i] ==1:
            yaSurvived+=1
    elif entry == 3:
        if survivalData.loc[i] == 0:
            aDeath+=1
        elif survivalData.loc[i] ==1:
            aSurvived+=1
    elif entry == 4:
        if survivalData.loc[i] == 0:
            sDeath+=1
        elif survivalData.loc[i] ==1:
            sSurvived+=1
    elif entry == -1:
        if survivalData.loc[i] == 0:
            uDeath+=1
        elif survivalData.loc[i] ==1:
            uSurvived+=1


deathTotals = (cDeath, tDeath, yaDeath, aDeath, sDeath, uDeath)
survivalTotals = (cSurvived, tSurvived, yaSurvived, aSurvived, sSurvived, uSurvived)

print(cDeath)

N=6
fig, ax = plt.subplots()
ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars
p1 = ax.bar(ind, survivalTotals, width, bottom=0)
p2 =  ax.bar(ind+width, deathTotals, width, bottom=0)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('C', 'T', 'YA', 'A', 'S', '?'))
ax.legend((p1[0], p2[0]), ('Survived', 'Death'))

# Save plot
filePath='..//plots//titanic//'
fileDir = os.path.dirname(os.path.realpath('__file__'))
plotPath = os.path.join(fileDir, filePath) #'../data/test.csv')
plotPath = os.path.abspath(os.path.realpath(plotPath))
plotName = 'AgeGroupSurvivalBarChart.png'
plt.savefig(os.path.join(plotPath, plotName), bbox_inches='tight')
#plt.show()


################################################################################

###############################################################################
# TRAIN DATA
fileName='train_cleaned.csv'
filePath = '..//data//titanic_cleaned//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
df = pd.read_csv(csvPath, sep=',', engine='python')
correlation = df.corr()
print(correlation)
