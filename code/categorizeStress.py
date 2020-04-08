import sys
import os
import pandas as pd

'''
Takes the csv-file containing the cleaned stressLvl entries (stressCleaned.csv)
and:
1. Calculates the average stress level based on the known entries
2. Replaces unknown entries with average stress level
3. Categorizes the stress levels:
        -  0-19 => low
        -  20-39 => med-low
        -  40-59 => med
        -  60-79 => med-high
        -  80-100 => high
4. Creates new stress level column where stress levels are replaced with
    corresponding category
5. Categorized stress level is saved to csv
    ..DM122/data/stressCategorized.csv
'''

fileName='stressCleaned.csv'
filePath = '..//data//' + fileName

# Get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# Read data
df = pd.read_csv(csvPath, sep=',', engine='python')
stressData = df['StressLvl']

# Calculate average stress level
count=0
sum=0
unknown_locs = []
for i, entry in enumerate(stressData):
    if 'unknown' in entry:
        unknown_locs.append(i)
    else:
        count+=1
        sum+= int(entry)
average=sum/count

# Replace unknowns with average stress level
for i in unknown_locs:
    stressData.loc[i]=average

# Categorize stress levels
for i, entry in enumerate(stressData):
    if int(entry) >= 80:
        stressData.loc[i] = 'high'
    elif int(entry)>=60:
        stressData.loc[i] = 'med-high'
    elif int(entry)>= 40:
        stressData.loc[i] = 'med'
    elif int(entry) >= 20:
        stressData.loc[i] = 'med-low'
    else:
        stressData.loc[i] = 'low'

for entry in stressData:
    print(entry)

# Save
csvName='stressCategorized.csv'
filePath = '..//data//' + csvName
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
stressData.to_csv(csvPath, header='StressLvl')
