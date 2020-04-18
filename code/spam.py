import pandas as pd
import string
import warnings
import nltk
import sklearn
from nltk import stem
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

i = 0
warnings.simplefilter(action='ignore', category=FutureWarning)
data = pd.read_csv("SmsCollection.csv", encoding = "UTF-8", delimiter= ';', error_bad_lines= False, warn_bad_lines=False)
stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))

#data processing
for sentence in data['text']:
    #remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    #remove stopwords & convert to lower case
    sentence = [word.lower() for word in sentence.split() if word not in stopwords]  
    #stemming
    sentence = " ".join([stemmer.stem(word) for word in sentence])
    #write back to the dataframe
    data['text'][i] = sentence
    i +=1

#print(data['text'][0])

#split dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size = 0.3, random_state = 42)

# training vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

# training the classifier 
svm = svm.SVC(C=1000)
svm.fit(X_train, y_train)

#training test set & predict
X_test = vectorizer.transform(X_test)
y_pred = svm.predict(X_test) 

#show results
print('\n')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred, labels = ['spam', 'ham']))
print('Accuracy = ', accuracy_score(y_test, y_pred))
print('Precision = ', precision_score(y_test, y_pred, pos_label= 'spam'))
print('Recall = ', recall_score(y_test, y_pred, pos_label= 'spam'))
print('F1 = ', f1_score(y_test, y_pred, pos_label= 'spam'))
print('\n\n\n')