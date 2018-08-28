# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 18:04:08 2018

@author: SBharadwaj
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# we use tsv and not csv because , the text may contain comma and this will confuse us
# so we use TSV(Tab). delimiter is to show what is the seperator
#  quoting =3 , will remove all quotes
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

import re
import nltk 
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#use regex to keep only the alphabets
ps=PorterStemmer()
X=[]
for i in range(0,dataset.shape[0]):
  review=re.sub("[^a-z A-Z]",'',string=dataset['Review'][i])
  review=review.lower()
  review=review.split()
  review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
  review=" ".join(review)
  X.append(review)
  
#lets create a big of words...
  from sklearn.feature_extraction.text import CountVectorizer
  cv=CountVectorizer(max_features=1500)
  X1=cv.fit_transform(X).toarray()
  
  y=dataset.iloc[:,1].values
# apply Calssification 
  #split data into training and test 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.2,random_state=0)


#Descision Tree
from sklearn.tree import DecisionTreeClassifier 
DTC=DecisionTreeClassifier(criterion="entropy",random_state=0)
DTC.fit(X_train,y_train)


#Predict 
y_pred=DTC.predict(X_test)

# compare results using confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

\



  

