import sys
import os
import pandas as pd

'''
Clean
'''

fileName='ODI-2020.csv'
filePath = '..//data//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
df = pd.read_csv(csvPath, sep=';', engine='python')

# change column names
print(df.columns)
df.columns = ['Programme', 'ML', 'IR', 'Statistics',
'Databases', 'Gender', 'Chocolate', 'Birthday', 'Neighbors', 'StandUp', 'StressLvl',
'Competition', 'RandomNr', 'BedTime', 'GoodDay1', 'GoodDay2']
print(df.columns)

# CLEAN PROGRAMME COLUMNS
progData = df['Programme']

# List of unique column entries
progList =[]
for entry in progData:
    if entry not in progList:
        progList.append(entry)

for entry in progList:
    print(entry)
print(len(progList))
print()

# Substrings for master programmes
CLSstring = ['Computational', 'computational']
AIstring = ['Artificial', 'AI']
QRMstring = ['Quantitative Risk', 'QRM', 'qrm', 'Quantitative risk',
    'quantitative risk']
BAstring = ['BA', 'Business Administration']
ECOstring = ['Econometrics', 'econometrics', 'EOR']
CSstring = ['Computer', 'computer', 'CS']
BIOstring = ['Bioinformatics']
BANstring = ['Business Analytics', 'Business analytics', 'business analytics']
FINstring = ['Finance', 'finance']
INFstring = ['Information']
DBIstring = ['Digital Business']
PAstring = ['Physics and Astronomy']
HLTstring = ['Human Language']
EXCstring = ['Exhange', 'exchange', 'Erasmus']
MSCstring = ['Master', 'MSc']

# Clean programme column
for i in progData.index:
    # CLS
    if any(substr in progData.loc[i] for substr in CLSstring):
        progData.loc[i] = 'CLS'
    # AI
    elif any(substr in progData.loc[i] for substr in AIstring):
        progData.loc[i] = 'AI'

    # QRM
    elif any(substr in progData.loc[i] for substr in QRMstring):
        progData.loc[i] = 'QRM'

    # Business Administration
    elif any(substr in progData.loc[i] for substr in BAstring):
        progData.loc[i] = 'BA'

    # Econometrics
    elif any(substr in progData.loc[i] for substr in ECOstring):
        progData.loc[i] = 'Econometrics'

    # Computer Science
    elif any(substr in progData.loc[i] for substr in CSstring):
        progData.loc[i] = 'CS'

    # Bioinformatics
    elif any(substr in progData.loc[i] for substr in BIOstring):
        progData.loc[i] = 'Bioinformatics'

    # Business Analytics
    elif any(substr in progData.loc[i] for substr in BANstring):
        progData.loc[i] = 'Business Analytics'

    # Finance
    elif any(substr in progData.loc[i] for substr in FINstring):
        progData.loc[i] = 'Finance'

    # Information Science/Information Studies
    elif any(substr in progData.loc[i] for substr in INFstring):
        progData.loc[i] = 'Information Science'

    # Digital Business and Innovation
    elif any(substr in progData.loc[i] for substr in DBIstring):
        progData.loc[i] = 'Digital Business and Innovation'

    # Physics and Astronomy
    elif any(substr in progData.loc[i] for substr in PAstring):
        progData.loc[i] = 'Physics and Astronomy'

    # Human Language Technology
    elif any(substr in progData.loc[i] for substr in HLTstring):
        progData.loc[i] = 'Human Language Technology'

    # Exchange
    elif any(substr in progData.loc[i] for substr in EXCstring):
        progData.loc[i] = 'Exchange'


    # Remove title & University
    progData.loc[i]=progData.loc[i].replace('Master ', '')
    progData.loc[i]=progData.loc[i].replace('MSc ', '')
    progData.loc[i]=progData.loc[i].replace(' UVA', '')

    if progData.loc[i] == 'MSc':
        progData.loc[i] = 'Unknown'


# List of unique column entries
progList=[]
for entry in progData:
    if entry not in progList:
        progList.append(entry)
for entry in progList:
    print(entry)

print(len(progList))
print()

# Save to csv
csvName='cleaned.csv'
filePath = '..//data//' + csvName
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
progData.to_csv(csvPath, header='Programme')
