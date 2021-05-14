# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:17:32 2021

@author: Josh
"""


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


#read training data

data = pd.read_csv("en_train.csv")

#np.where(pd.isnull(data['before'])) # to check null values

data_exp=data.fillna(" ")

np.where(pd.isnull(data_exp['before']))

X = data_exp[['before']]
y = data_exp[['class']]


count_vect = CountVectorizer()  # pre-process content as string to float and then split

X = count_vect.fit_transform(data_exp['before'])

#print(X)

#print(data(X.A, columns=data.get_feature_names()).to_string())

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3) 

#manually split the data into train test split

X_train = X[0:700000]
X_test = X[700000:1048575]
y_train = y[0:700000]
y_test = y[700000:1048575]

################################# Run Naive Bayes

mnb = MultinomialNB()
mnb.fit(X_train, y_train.values.ravel())

#gnb = GaussianNB()
#gnb.fit(X_train, y_train.values.reshape(-1,1))

y_pred = mnb.predict(X_test)


y_pred_list = y_pred.tolist()

new_data_exp = data_exp[700000:1048575]

new_data_exp['predicted_class'] = y_pred_list

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


##################################### Run SVM
 
from sklearn.svm import LinearSVC
linearsvc = LinearSVC()
linearsvc.fit(X_train,y_train.values.ravel())


y_pred = linearsvc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))




######################## #CREATE THE AFTER PREDICTION



import inflect 
import regex as re 
inflector = inflect.engine()

import re

instring = "TEST"
lower_words = instring.lower()

 # counter from here https://stackoverflow.com/questions/18129830/count-the-uppercase-letters-in-a-string-with-python
 #count = len([letter for letter in instring if letter.isupper()])




predicted_after_list = []


for index, row in new_data_exp.iterrows():
    if row['predicted_class'] == 'CARDINAL':
        words = inflector.number_to_words(row['before'])
        #words = words.replace("and", '')
        words = re.sub('[!@#$-]', ' ', words)
        #print(words)
        predicted_after_list.append(words)
    elif row['predicted_class'] == 'LETTERS':
        words = row['before']
        count_upper = len([letter for letter in words if letter.isupper()])
        if count_upper > 2:
            words = re.sub('[!@#$-.]', ' ', words)
            words = words.lower()
            words = " ".join(words)
            #print(words)
        predicted_after_list.append(words)
            
    else:
        predicted_after_list.append(row['before'])
        
new_data_exp['predicted_after'] = predicted_after_list

match_list = []

for index, row in new_data_exp.iterrows():
    if row['predicted_after'] == row['after']:
        match_list.append('yes')
    else:
        match_list.append('no')

new_data_exp['match'] = match_list

#Percentage match - the length of the 

accuracy = len(new_data_exp[new_data_exp.match == 'yes']) / len(new_data_exp)



