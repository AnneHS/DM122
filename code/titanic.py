import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import math

'''
Creates new cleaned dataset
- PassengerId: int
- Survived: yes:1, no:0
- Pclass: first:1, middle:2, lower:3
- Sex: female:1, male:0
- Age: float
- Agegroup (age): child:0, teenager:1, young adult:2, adult:3, senior:4
- Deck (cabin): A:0, B:1, C:2, ..., G:6, T:7
- Number (cabin): int
- Side (cabin): starboard:1, port side:2
- Embarked: Cherbourg:1, Queenstown:2, Southampton:3

* all unknowns set to -1
'''

fileName='train.csv'
filePath = '..//data//titanic//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
df = pd.read_csv(csvPath, sep=',', engine='python')


###############################################################################
# SEX
###############################################################################

sexData=df['Sex']
sex=[]
sex_dict={'male':0, 'female':1}
for entry in sexData:
    sex.append(sex_dict[entry])



###############################################################################
# Title
###############################################################################

#TODO: Replace various titles with Dr, Mr, Mrs (NEED TO COMBINE WITH TEST DATA FIRST)


###############################################################################
# AGE
###############################################################################

ageData = df['Age']
unique=[]

# Get row indices of missing entries
loc = np.where(pd.isnull(ageData))
unknown_index=loc[0]

age=[]
ageGroup=[]
for i, entry in enumerate(ageData):
    if i in unknown_index:
        age.append(-1)
        ageGroup.append(-1)
    else:
        age.append(entry)

        if entry < 14:
            ageGroup.append(0)  # Child
        elif entry < 25:
            ageGroup.append(1)  # Teenager
        elif entry < 35:
            ageGroup.append(2)  # Young adult
        elif entry < 60:
            ageGroup.append(3)  # Adult
        else:
            ageGroup.append(4)  # Senior



###############################################################################
# FARE
###############################################################################

#TODO: Replace missing ticket fares with average for their passenger class

###############################################################################
# CABIN
##############################################################################

# Get cabin data
cabinData = df['Cabin']

# Get row indices of missing entries
loc = np.where(pd.isnull(cabinData))
rows=loc[0]

# Assign numbers to deck letters
deck_dict={'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7}

# port side = left(even) strictly enforced
# starboard = right (uneven) not strictly enforced

# Three arrays: decks, room number, side (starboard/port side)
decks=[]
numbers=[]
sides=[]
for i, entry in enumerate(cabinData):
    if i in rows:
        decks.append(-1)
        numbers.append(-1)
        sides.append(-1)
    else:
        cabins = entry.split()

        # One cabin
        if len(cabins)==1:

            # Get deck (A, B, C,...)
            d=deck_dict[cabins[0][0]]
            decks.append(d)

            # If cabin number known
            if len(cabins[0])>1:

                # Get number
                numbers.append(cabins[0][1:])

                # Get side:  starboard (uneven), port side (even)
                if int(cabins[0][-1])%2 ==0:
                    sides.append(2)
                else:
                    sides.append(1)

            else:
                numbers.append(-1)
                sides.append(-1)
        else:
            ds=[]
            ns=[]
            s=[]
            for cabin in cabins:

                # Get deck (A, B, C...)
                ds.append(deck_dict[cabin[0]])

                # if number known
                if len(cabin)>1:
                    ns.append(int(cabin[1:]))

                    # Get side: port side (even), starboard (uneven)
                    if int(cabin[-1])%2==0:
                        s.append(2)
                    else:
                        s.append(1)
                else:
                    ns.append(-1)

            decks.append(ds)
            numbers.append(ns)
            sides.append(s)





##############################################################################
# EMBARKED
##############################################################################
embarkedData = df['Embarked']

unique=[]
for entry in embarkedData:

    if entry not in unique:
        unique.append(entry)

# Get row indices of missing entries
loc = np.where(pd.isnull(embarkedData))
unknown_index=loc[0]

# Number categories
embarked=[]
embarkedDict={'C':1, 'Q': 2, 'S': 3}
for i, entry in enumerate(embarkedData):

    # Unknown => -1
    if i in unknown_index:
        embarked.append(-1)
    else:
        embarked.append(embarkedDict[entry])
print(embarked)



###############################################################################
# CLEANED => DATAFRAME => CSV
##############################################################################
# New Dataframe
d={'Survived': df['Survived'], 'Pclass': df['Pclass'], 'Sex': sex, 'Age': age,
    'AgeGroup': ageGroup, 'Deck':decks, 'Number': numbers, 'Side': sides,
    'Embarked': embarked}
cleaned_df = pd.DataFrame(d)
cleaned_df.index.name='PassengerId'

# Save to csv
csvName='titanic_cleaned.csv'
filePath = '..//data//titanic_cleaned//' + csvName
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
cleaned_df.to_csv(csvPath)



###############################################################################
# BAR CHARTS
###############################################################################
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
