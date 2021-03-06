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
- FamSize: int (SibSp + Parch + 1)
- isAlone: yes:1, no:0
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
title=title.replace(['Mlle','Ms','Countess','Miss','Mme', 'Lady'], 'Mrs')
title=title.replace(['Major','Sir','Capt','Col','Don','Jonkheer','Rev','Master'], 'Mr')

# Mapping
title_mapping={'Dr':0, 'Mr':1, 'Mrs':2}
title=title.map(title_mapping)
title=title.fillna(-1)


# TEST DATA
testName=testdf['Name']

# Replace various titles with more common title
titleTest=testName.str.extract(' ([A-Za-z]+)\.', expand=False)
titleTest=titleTest.replace(['Mlle','Ms','Countess','Miss','Mme', 'Lady'], 'Mrs')
titleTest=titleTest.replace(['Major','Sir','Capt','Col','Don','Jonkheer','Rev','Master'], 'Mr')

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
df['Sex']=sex

# TEST DATA
sexTestData=testdf['Sex']
sexTest=sexTestData.map(sex_mapping)




###############################################################################
# AGE => Missing values
###############################################################################

df['Salutation']=df.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
grp=df.groupby(['Sex', 'Pclass'])
df.Age=grp.Age.apply(lambda x: x.fillna(x.median()))
df.Age.fillna(df.Age.median, inplace = True)
#ageData = df.Age

testdf['Salutation']=testdf.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
grp=testdf.groupby(['Sex', 'Pclass'])
testdf.Age=grp.Age.apply(lambda x: x.fillna(x.median()))
testdf.Age.fillna(testdf.Age.median, inplace = True)
#ageData = df.Age


###############################################################################
# AGEGROUP (AGE)
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

count=0
for entry in ageGroup:
    if entry==-1:
        count+=1
#print(count)

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
# FAMILY SIZE & ISALONE (PARCH & SIBSP)
##############################################################################

# TRAIN
sibspData=df['SibSp']
parchData=df['Parch']
isAlone=(sibspData + parchData).apply(lambda x: 0 if x>0 else 1)
famSize=sibspData+parchData+1

#TEST
sibspData=testdf['SibSp']
parchData=testdf['Parch']
isAloneTest=(sibspData + parchData).apply(lambda x: 0 if x>0 else 1)
famSizeTest=sibspData+parchData+1



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
cabinBool=[]
for i, entry in enumerate(cabinData):
    if i in rows:
        decks.append(-1)
        numbers.append(-1)
        sides.append(-1)
        cabinBool.append(0)
    else:
        cabinBool.append(1)
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
cabinBoolTest=[]
for i, entry in enumerate(cabinTestData):
    if i in rows:
        decksTest.append(-1)
        numbersTest.append(-1)
        sidesTest.append(-1)
        cabinBoolTest.append(0)
    else:
        cabinBoolTest.append(1)
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
embarked=embarked.fillna(-1)
#print(embarked)

# TEST
embarkedTestData = testdf['Embarked']
embarkedTest=embarkedTestData.map(embarked_mapping)
embarkedTest=embarkedTest.fillna(-1)
#print(embarkedTest)



###############################################################################
# CLEANED => DATAFRAME => CSV
##############################################################################

# TRAINING SET
# New Dataframe
d={'Survived': df['Survived'], 'Pclass': df['Pclass'], 'Title': title,
    'Surname': surnames,'Sex': sex, 'Age': df['Age'], 'SibSp': df['SibSp'],
    'Parch': df['Parch'], 'FamSize':famSize, 'isAlone':isAlone,
    'Fare': fareData, 'TktNum': TktNum,'Embarked': embarked,
    'CabinBool': cabinBool,'Deck': decks,'CabinNumber': numbers, 'Side': sides,
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
    'Sex': sexTest,'Age': testdf['Age'], 'SibSp': testdf['SibSp'],
    'Parch': testdf['Parch'], 'FamSize': famSizeTest, 'isAlone':isAloneTest,
    'TktNum': TktNumTest, 'Fare': fareTestData,'Embarked': embarkedTest,
    'CabinBool': cabinBoolTest, 'Deck': decksTest, 'CabinNumber': numbersTest,
    'Side': sidesTest,'AgeGroup': ageGroupTest, 'FareGroup': fareTestGrouped}
cleaned_test=pd.DataFrame(dTest)
cleaned_test.index.name='PassengerId'

# Save to csv
csvName='test_cleaned.csv'
filePath = '..//data//titanic_cleaned//' + csvName
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
cleaned_test.to_csv(csvPath)
