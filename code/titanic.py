import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import math

'''
Cleans train.csv and test.csv => train_cleaned.csv, test_cleaned.csv
- PassengerId: int
- Survived*: no:0, yes:1
- Pclass: first: 1, middle:2, lower:3
- Title (Name): Dr:0, Mr:1, Mrs:2
- Sex: male:0, female:1
- Age: float
- SibSp:  int (# of siblings/spoused aboard)
- Parch: int (# of parents/children aboard)
- Fare**: float (ticket price)
- Embarked: Cherbourg:1, Queenstown:2, Southampton:3
- Deck (Cabin): A:0, B:1, C:2, ..., G:6, T:7
- CabinNumber (Cabin): int
- Side (Cabin): starboard:1, port side:2
- AgeGrouped: child:0, teenager:1, young adult:2, adult:3, senior:4
- FareGrouped***: 1st:0, 2nd:1, 3rd:2, 4rd:3
* Survived only for train_cleaned.csv
** Uknown entries were replaced with average fare for Pclass
*** Fares grouped into 4 quartiles (equally sized)

All unknowns are set to -1
'''

###############################################################################
# DATA
###############################################################################

# TRAIN DATA
fileName='train.csv'
filePath = '..//data//titanic//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
df = pd.read_csv(csvPath, sep=',', engine='python')



# TEST DATA
fileName='test.csv'
filePath = '..//data//titanic//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
testdf = pd.read_csv(csvPath, sep=',', engine='python')

###############################################################################
# TITLE (from NAME)
###############################################################################

# TRAIN DATA
dataName=df['Name']

# Replace various titles with more common title
title=dataName.str.extract(' ([A-Za-z]+)\.', expand=False)
title=title.replace(['Mlle','Ms','Countess','Miss','Mme'], 'Mrs')
title=title.replace(['Major','Sir','Capt','Col','Don','Jonkheer','Rev','Master','Lady'], 'Mr')

# Mapping
title_mapping={'Dr':0, 'Mr':1, 'Mrs':2}
title=title.map(title_mapping)
title=title.fillna(-1)



# TEST DATA
testName=testdf['Name']

# Replace various titles with more common title
titleTest=testName.str.extract(' ([A-Za-z]+)\.', expand=False)
titleTest=titleTest.replace(['Mlle','Ms','Countess','Miss','Mme'], 'Mrs')
titleTest=titleTest.replace(['Major','Sir','Capt','Col','Don','Jonkheer','Rev','Master','Lady'], 'Mr')

# Mapping
titleTest=titleTest.map(title_mapping)
titleTest=titleTest.fillna(-1)


###############################################################################
# SURNAME (FROM NAME)
###############################################################################

surnames=[]
for entry in dataName:
    LastFirst = entry.split(',')
    surnames.append(LastFirst[0])

surnamesTest=[]
for entry in testName:
    LastFirst = entry.split(',')
    surnamesTest.append(LastFirst[0])




###############################################################################
# SEX
###############################################################################

# TRAIN DATA
sexData=df['Sex']
sex_mapping={'male':0, 'female':1}
sex=sexData.map(sex_mapping)


# TEST DATA
sexTestData=testdf['Sex']
sexTest=sexTestData.map(sex_mapping)



###############################################################################
# AGE
###############################################################################

# TRAIN DATA
ageData = df['Age']

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


# TEST DATA
ageTestData = testdf['Age']

# Get row indices of missing entries
loc = np.where(pd.isnull(ageTestData))
unknown_index=loc[0]

ageTest=[]
ageGroupTest=[]
for i, entry in enumerate(ageTestData):
    if i in unknown_index:
        ageTest.append(-1)
        ageGroupTest.append(-1)
    else:
        ageTest.append(entry)

        if entry < 14:
            ageGroupTest.append(0)  # Child
        elif entry < 25:
            ageGroupTest.append(1)  # Teenager
        elif entry < 35:
            ageGroupTest.append(2)  # Young adult
        elif entry < 60:
            ageGroupTest.append(3)  # Adult
        else:
            ageGroupTest.append(4)  # Senior



##############################################################################
# TktNum FROM TICKET
##############################################################################

# TRAIN
ticketData = df['Ticket']
#TktPre=[]
TktNum=[]
unique=[]
for entry in ticketData:
    PreNum=entry.split()
    if PreNum[-1]== 'LINE':
        TktNum.append(-1)
    else:
        TktNum.append(PreNum[-1])


#TEST
ticketDataTest = testdf['Ticket']
#TktPreTest=[]
TktNumTest=[]
for entry in ticketDataTest:
    PreNum=entry.split()
    if PreNum[-1]== 'LINE':
        TktNumTest.append(-1)
    else:
        TktNumTest.append(PreNum[-1])

#TODO: TktPre???



###############################################################################
# FARE
###############################################################################

# TRAIN (has no missing entries)
fareData = df['Fare']
'''
loc = np.where(pd.isnull(fareData))
rows=loc[0]
print(rows)
'''

# TEST:  Replace missing ticket fares with average for their passenger class
fareTestData = testdf['Fare']
for x in range(len(fareTestData)):
     if pd.isnull(fareTestData[x]):
        pclass = testdf["Pclass"][x]
        fareTestData[x] = round(fareData[df["Pclass"] == pclass].mean(), 4)

# CATEGORIZE FARE
fareGrouped = pd.qcut(fareData, 4, labels = [0, 1, 2, 3])
fareTestGrouped = pd.qcut(fareTestData, 4, labels = [0, 1, 2, 3])



###############################################################################
# CABIN: DECK, NUMBER, SIDE
##############################################################################

#TRAIN DATA
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


#TEST DATA
# Get cabin data
cabinTestData = testdf['Cabin']

# Get row indices of missing entries
loc = np.where(pd.isnull(cabinTestData))
rows=loc[0]

# Three arrays: decks, room number, side (starboard/port side)
decksTest=[]
numbersTest=[]
sidesTest=[]
for i, entry in enumerate(cabinTestData):
    if i in rows:
        decksTest.append(-1)
        numbersTest.append(-1)
        sidesTest.append(-1)
    else:
        cabins = entry.split()

        # One cabin
        if len(cabins)==1:

            # Get deck (A, B, C,...)
            d=deck_dict[cabins[0][0]]
            decksTest.append(d)

            # If cabin number known
            if len(cabins[0])>1:

                # Get number
                numbersTest.append(cabins[0][1:])

                # Get side:  starboard (uneven), port side (even)
                if int(cabins[0][-1])%2 ==0:
                    sidesTest.append(2)
                else:
                    sidesTest.append(1)

            else:
                numbersTest.append(-1)
                sidesTest.append(-1)
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

            decksTest.append(ds)
            numbersTest.append(ns)
            sidesTest.append(s)

##############################################################################
# EMBARKED
##############################################################################

embarked_mapping={'C':1, 'Q': 2, 'S': 3}

# TRAIN
embarkedData = df['Embarked']
embarked=embarkedData.map(embarked_mapping)
embarked=embarkedData.fillna(-1)

# TEST
embarkedTestData = testdf['Embarked']
embarkedTest=embarkedTestData.map(embarked_mapping)
embarkedTest=embarkedTest.fillna(-1)


###############################################################################
# CLEANED => DATAFRAME => CSV
##############################################################################

# TRAINING SET
# New Dataframe
d={'Survived': df['Survived'], 'Pclass': df['Pclass'], 'Title': title,
    'Surname': surnames,'Sex': sex, 'Age': age, 'SibSp': df['SibSp'],
    'Parch': df['Parch'], 'Fare': fareData, 'TktNum': TktNum,
    'Embarked': embarked, 'Deck': decks,'CabinNumber': numbers, 'Side': sides,
    'AgeGroup': ageGroup,'FareGroup':fareGrouped}
cleaned_df = pd.DataFrame(d)
cleaned_df.index.name='PassengerId'

# Save to csv
csvName='train_cleaned.csv'
filePath = '..//data//titanic_cleaned//' + csvName
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
cleaned_df.to_csv(csvPath)

# TESTING SET
# New DataFrame
dTest={'Pclass': testdf['Pclass'], 'Title': titleTest, 'Surname':surnamesTest,
    'Sex': sexTest,'Age': ageTest, 'SubSp': testdf['SibSp'],
    'Parch': testdf['Parch'],'TktNum': TktNumTest, 'Fare': fareTestData,
    'Embarked': embarkedTest,'Deck': decksTest, 'CabinNumber': numbersTest,
    'Side': sidesTest,'AgeGroup': ageGroupTest, 'FareGroup': fareTestGrouped}
cleaned_test=pd.DataFrame(dTest)
cleaned_test.index.name='PassengerId'

# Save to csv
csvName='test_cleaned.csv'
filePath = '..//data//titanic_cleaned//' + csvName
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
cleaned_test.to_csv(csvPath)





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
