import sys
import numpy as np
import pandas as pd
import scipy as sp
import sklearn

stopwords = set(['a', 'ako', 'ali', 'od', 'da', 'do', 'ka', 'u', 'za', 'jer', 'je', 'jeste', 'na', 'onaj', 'ona', 'ono', 'onog', 'onom', 'onim', 'one', 'onoj', 'onu', 'koji', 'koja', 'koje', 'taj', 'ta', 'to', 'tog', 'tom', 'tome', 'tim', 'te', 'toj', 'tu', 'takav', 'takva', 'takvo', 'takvog', 'takvom', 'takvim', 'takve', 'takvoj', 'takvu', 'onakav', 'onakva', 'onakvo', 'onakvog', 'onakvom', 'onakvim', 'onakve', 'onakvoj', 'onakvu', 'toliki', 'tolika', 'toliko', 'tolikog', 'tolikom', 'tolikim', 'tolike', 'tolikoj', 'toliku', 'onoliki', 'onolika', 'onoliko', 'onolikog', 'onolikom', 'onolikim', 'onolike', 'onolikoj', 'onoliku', 'sa', 'ovaj', 'ova', 'ovo', 'ovog', 'ovom', 'ovim', 'ove', 'ovoj', 'ovu', 'ovakav', 'ovakva', 'ovakvo', 'ovakvog', 'ovakvom', 'ovakvim', 'ovakve', 'ovakvoj', 'ovoliki', 'ovolika', 'ovoliko', 'ovolikog', 'ovolikom', 'ovolikim', 'ovolike', 'ovolikoj', 'ovoliku', 'biti', 'je', 'jesam', 'sam', 'si', 'smo', 'ste', 'su', 'jesi', 'jesmo', 'jeste', 'jesu', 'kao', 'ja', 'iz', 'neki', 'neka', 'neko', 'nekog', 'nekom', 'nekim', 'neke', 'nekoj', 'jedan', 'jedna', 'jedno', 'jednog', 'jednom', 'jednim', 'jedne', 'jednoj', 'jednu', 'neki', 'neka', 'neko', 'nekog', 'nekom', 'nekim', 'neke', 'nekoj', 'jedan', 'jedna', 'jedno', 'jednog', 'jednom', 'jednim', 'jedne', 'jednoj', 'jednu', 'o', 'oko', 'zbog', 'skoro', 'gotovo', 'bezmalo', 'sada', 'odmah', 'baš', 'bas', 'upravo', 'spreman', 'bio', 'bila', 'bilo', 'ko', 'koji', 'koja', 'koje', 'šta', 'sta', 'kakav', 'kakva', 'kakvo', 'kakvog', 'kakvom', 'kakvim', 'kakve', 'kakvoj', 'kakvu', 'koliki', 'kolika', 'koliko', 'kolike', 'gde', 'kuda', 'odakle', 'otkuda', 'kada', 'kad', 'čim', 'cim', 'zašto', 'zasto', 'što', 'sto', 'kako', 'koji', 'koja', 'koje', 'kojeg', 'koga', 'kojem', 'kojim', 'koje', 'kojoj', 'koju', 'kojom', 'i', 'takođe', 'takodje', 'još', 'jos', 'to', 'ono', 'na', 'u', 'ka', 'kod', 'oko', 'pored', 'pri', 'ili'])

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

train_set_path = sys.argv[1]
test_set_path = sys.argv[2]

train_data = pd.read_json(train_set_path)
test_data = pd.read_json(test_set_path)

train_data['strofa'] = train_data['strofa'].apply(preprocess_text)
test_data['strofa'] = test_data['strofa'].apply(preprocess_text)

X_train = train_data['strofa']
X_test = test_data['strofa']

print(X_train.count)

y_train = train_data['zanr']
y_test = test_data['zanr']

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(train_data['strofa'], train_data['zanr'], test_size=0.3, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

classifier = SVC(C = 1.0, kernel = 'rbf')
classifier.fit(X_train_tfidf, y_train)

y_pred = classifier.predict(X_test_tfidf)

micro_f1 = f1_score(y_test, y_pred, average='micro')
print(micro_f1)


