import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

'''
Cleans various columns of ODI-2020.csv and creates corresponding plots.

Master Programme:
1. Inventarisation of unique entries in Master Programme column.
2. Picks one term/name for each unique master programme and replaces all
    entries using a different term for referring to that same programme,
    with this term (e.g. AI for all entries referring to Artificial Intelligence)
3. Creates and saves Pie chart
    ...DM122/plots/mscProgrammesPieChart.png
4. 'Cleaned' data is saved to csv
    ...DM122/plots/mscCleaned
* Information Studies and Information Science merged into one category;
Information Science

Good Day 1 & 2
1. Extracts all entries from the 2 'What makes a good day' columns.
2. Creates and saves a Word Chart based on these entries
    ...DM122/data/GoodDatWordChart.png

Stress level
Extracts Stress level column and saves in seperate csv-file
    ...DM122/data/stressCleaned.csv
* Actual cleaning happened manually
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

    elif 'Human movement' in progData.loc[i]:
        progData.loc[i] = 'Human Movement'

    elif 'medical informatics'in progData.loc[i]:
        progData.loc[i] = 'Medical Informatics'

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
progDict={k: v for k, v in sorted(progDict.items(), key=lambda item: item[1], reverse=True)}
# Plot pie chart
# Documentation: https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_and_donut_labels.html
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(aspect="equal"))

labels = [(str(prog) + " (" + str(progDict[prog]) + ")") for prog in progDict]
data = [amount for amount in progDict.values()]

#clmap=plt.get_cmap("tab20b")
#colors= clmap(range(20))
NUM_COLORS=25
cm=plt.get_cmap('gist_ncar')
colors=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
#for i in range(NUM_COLORS):
#    colors.append(cm(i//3*3.0/NUM_COLORS))
wedges, texts = ax.pie(data, colors=colors, wedgeprops=dict(width=0.5), startangle=-40)

for w in wedges:
    w.set_linewidth(1)
    w.set_edgecolor('white')
ax.legend(wedges, labels,
          title="Programmes",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

'''
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
'''


# Save pie chart => .../DM122/plots
filePath='..//plots//'
fileDir = os.path.dirname(os.path.realpath('__file__'))
plotPath = os.path.join(fileDir, filePath) #'../data/test.csv')
plotPath = os.path.abspath(os.path.realpath(plotPath))
plotName = 'MscProgrammesPieChart.png'
plt.savefig(os.path.join(plotPath, plotName), bbox_inches='tight')
plt.show()

# Save cleaned data to csv => .../DM122/data
csvName='mscCleaned.csv'
filePath = '..//data//' + csvName
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
progData.to_csv(csvPath, header='Programme')



# What makes a good day?
gd1Data = df['GoodDay1']
gd2Data = df['GoodDay2']

dayList =[]
for entry in gd1Data:
    if entry not in dayList:
        dayList.append(entry)
for entry in gd2Data:
    if entry not in dayList:
        dayList.append(entry)

for entry in dayList:
    print(entry)
print(len(dayList))
print()

# Concatenate entries
text1 = " ".join(entry for entry in gd1Data)
text2 = " ".join(entry for entry in gd2Data)
text = text1 + text2

# Create and generate a word cloud image
wordcloud = WordCloud(width=800, height=400, min_font_size=15, background_color='white', colormap='tab20').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# Save
plotName = 'GoodDayWordChart.png'
plt.savefig(os.path.join(plotPath, plotName), bbox_inches='tight')
plt.show()

stressData = df['StressLvl']
csvName='stressCleaned.csv'
filePath = '..//data//' + csvName
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
#stressData.to_csv(csvPath, header='Programme')



#DIFFERENT COURSES (currently not used)
mlData = df['ML']
irData = df['IR']
statData = df['Statistics']
dbData = df['Databases']

n=len(mlData)
init = np.zeros((n))
d={'Courses': pd.Series(init)}
courseDF = pd.DataFrame(d)

for i, entry in enumerate(mlData):
    if entry == 'yes':
        courseDF.loc[i] +=1
        mlData.loc[i] = 1
    else:
        mlData.loc[i] =0

for i, entry in enumerate(irData):
    if entry == 1:
        courseDF.loc[i] +=1
    else:
        irData.loc[i] =0
for i, entry in enumerate(statData):
    if entry == 'mu':
        courseDF.loc[i] +=1
        statData.loc[i] = 1
    else:
        statData.loc[i] =0

for i, entry in enumerate(dbData):
    if entry == 'ja':
        courseDF.loc[i] +=1
        dbData.loc[i] = 1
    else:
        dbData.loc[i] =0
print(courseDF)

fileNameRegression='dataset_for_regression.csv'
filePath = '..//data//' + fileNameRegression

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))

# read data
regData = pd.read_csv(csvPath, engine='python')
regData['Machine Learning'] = mlData
regData['Information Retrieval'] = irData
regData['Statistics'] = statData
regData['Databases'] = dbData
del regData['Num courses']
regData = regData[['Gender', 'Year of Birth', 'Bed Time', 'Machine Learning',
    'Information Retrieval', 'Statistics', 'Databases', 'StressLvl']]

csvName='regression_with_courses.csv'
filePath = '..//data//' + csvName
fileDir = os.path.dirname(os.path.realpath('__file__'))
csvPath = os.path.join(fileDir, filePath)
csvPath = os.path.abspath(os.path.realpath(csvPath))
regData.to_csv(csvPath, header='Programme')


# Print nondigit answers for competition question
count=0
compData = df['Competition']
for i, entry in enumerate(compData):
    if not entry.isdigit():
        print(entry)
        courseDF.drop(i)
        compData.drop(labels=i)
        count+=1
print(count)


# Adding code for bed time vs gender and distribution of random numbers

#copying this bit to get the path
fileName='ODI-2020.xlsx'
filePath = '..//data//' + fileName

# get path
fileDir = os.path.dirname(os.path.realpath('__file__'))
excelpath = os.path.join(fileDir, filePath)
excelpath = os.path.abspath(os.path.realpath(csvPath))

df = pd.read_excel (r'C:\Users\39331\Desktop\UvA\Data Mining\hw\project 1\DM122\data\ODI-2020.xlsx')

#bed time vs gender
plt.close()
gend = df.iloc[:, [5,13]]
gend.iloc[:,1] = gend.iloc[:,1].replace({':':'.'}, regex=True)
gend.iloc[:,1] = gend.iloc[:,1].replace({'am':''}, regex=True)
gend.iloc[:,1] = gend.iloc[:,1].replace({'AM':''}, regex=True)


for i in range(len(gend)):
    if type(gend.iloc[i,1])== str and 'pm' in gend.iloc[i,1]:
            gend.iloc[i,1] = gend.iloc[i,1].replace('pm','')
            gend.iloc[i,1] = float(gend.iloc[i,1])
            gend.iloc[i,1] = gend.iloc[i,1] + 12.0

for i in range(len(gend)):
    if type(gend.iloc[i,1])== int or type(gend.iloc[i,1])== float and gend.iloc[i,1]>99:
        gend.iloc[i,1] = gend.iloc[i,1] /100.0
    if type(gend.iloc[i,1])== int and gend.iloc[i,1]>24 and gend.iloc[i,1]<0:
        gend.iloc[i,1] = np.nan
    if type(gend.iloc[i,1]) == str:
        gend.iloc[i:,1] = pd.to_numeric(gend.iloc[i:,1], downcast="float", errors='coerce')
        if gend.iloc[i,1]>99:
            gend.iloc[i,1] = gend.iloc[i,1] /100.0

gend.iloc[:,1] = pd.to_numeric(gend.iloc[:,1], downcast="float", errors='coerce')


bedt=gend.iloc[:,1].values
gender = gend.iloc[:,0].values


male = []
female = []
unknown = []

for i in range(len(gender)):
    if gender[i] == 'male' and bedt[i] <25:
        male.append(bedt[i])
    elif gender[i]== 'female' and bedt[i]<25:
        female.append(bedt[i])
    elif bedt[i]<25:
         unknown.append(bedt[i])

bins = np.linspace(0, 24, 1)
plt.hist([male, female], bins=24, density=True, label = ['male', 'female'])
plt.legend()
my_xticks = ['00:00','2:00','4:00','6:00', "8:00", "10:00", '12:00','14:00','16:00', "18:00", "20:00", "22:00" ]
plt.xlim(0,24)
plt.xticks(np.arange(0,24,2), my_xticks, rotation=45)
plt.xlabel("Bed Time", fontsize = 13)
plt.ylabel("Percentage of Students by Gender", fontsize = 13)


# Save pie chart => .../DM122/plots
filePath='..//plots//'
fileDir = os.path.dirname(os.path.realpath('__file__'))
plotPath = os.path.join(fileDir, filePath) #'../data/test.csv')
plotPath = os.path.abspath(os.path.realpath(plotPath))
plotName = "gender_bed_times.png"
plt.savefig(os.path.join(plotPath, plotName))
plt.close()

#PLOT FOR DISTRIBUTION OF RANDOM NUMBERS

random = df.iloc[:, 12]

random = pd.to_numeric(random, downcast="float", errors='coerce')

random=random.values
count = np.zeros(4)

for i in range(len(random)):
    if random[i]>1000:
        count[3] += 1
        random[i] = np.nan
    elif random[i]>100:
        count[2] += 1
    elif random[i]>10:
        count[1] += 1
    else:
        count[0] += 1

bins=np.arange(-1,1000,1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.hist(random, bins)
ax1.set_xlim(-5, 105)
ax1.set_xlabel('Random number selected', fontsize=13)
ax1.set_ylabel("Frequency", fontsize=13)
ax2.hist(random, bins)
ax2.set_xlim(-5, 25)
ax2.set_xlabel('Random number selected', fontsize=13)
ax2.set_ylabel("Frequency", fontsize=13)
plt.tight_layout()
plotName = "distribution_rand_numbers.png"
plt.savefig(os.path.join(plotPath, plotName))
plt.close()

print('jo')
