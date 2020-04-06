import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

'''
Clean

For the Msc Programmes column
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
#print(df.columns)
df.columns = ['Programme', 'ML', 'IR', 'Statistics',
'Databases', 'Gender', 'Chocolate', 'Birthday', 'Neighbors', 'StandUp', 'StressLvl',
'Competition', 'RandomNr', 'BedTime', 'GoodDay1', 'GoodDay2']
#print(df.columns)

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

    # Mechanical Engineering
    elif 'Mechenaical' in progData.loc[i]:
        progData.loc[i] = 'Mechanical Engineering'

    # Exchange
    elif any(substr in progData.loc[i] for substr in EXCstring):
        progData.loc[i] = 'Exchange'

    # Unknown
    if progData.loc[i] == 'MSc':
        progData.loc[i] = 'Unknown'

    # Remove title & University
    progData.loc[i]=progData.loc[i].replace('Master ', '')
    progData.loc[i]=progData.loc[i].replace('MSc ', '')
    progData.loc[i]=progData.loc[i].replace(' UVA', '')


# Dictionary (key=programme, value=nr of students)
progDict={}
for entry in progData:
    if entry in progDict:
        progDict[entry]+=1
    else:
        progDict[entry]=1

# Plot pie chart
# Documentation: https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_and_donut_labels.html
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(aspect="equal"))

labels = [(str(prog) + " (" + str(progDict[prog]) + ")") for prog in progDict]
data = [amount for amount in progDict.values()]

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw, size=14)

# Save pie chart => .../DM122/plots
filePath='..//plots//'
fileDir = os.path.dirname(os.path.realpath('__file__'))
plotPath = os.path.join(fileDir, filePath) #'../data/test.csv')
plotPath = os.path.abspath(os.path.realpath(plotPath))
plotName = 'MscProgrammesPieChart.png'
plt.savefig(os.path.join(plotPath, plotName))
plt.show()

# Save cleaned data to csv => .../DM122/data
csvName='cleaned.csv'
filePath = '..//data//' + csvName
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
progData.to_csv(csvPath, header='Programme')



# What makes a good day?
gd1Data = df['GoodDay1']
gd2Data = df['GoodDay2']

# Concatenate entries
text1 = " ".join(entry for entry in gd1Data)
text2 = " ".join(entry for entry in gd2Data)
text = text1 + text2

# Create and generate a word cloud image:
wordcloud = WordCloud(width=800, height=400, colormap='Pastel1').generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# Save
plotName = 'GoodDayWordChart.png'
plt.savefig(os.path.join(plotPath, plotName))
plt.show()
