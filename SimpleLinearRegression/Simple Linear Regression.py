#simple linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


os.getcwd()
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,1]

#Splitting the dataset into the Training set and Test set 
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

# Creat the Simple Linear Regression 
from sklearn.linear_model import LinearRegression
LRmodel=LinearRegression()
LRmodel.fit(X_train,y_train)

#predict using the Test data
y_pred=LRmodel.predict(X_test)

#Visualize the training data
plt.scatter(X_train,y_train,color='red')
#Visulaize the test data but remeber to keep the same plot line 
plt.scatter(X_test,y_test,color='black')
plt.plot(X_train,LRmodel.predict(X_train),color='blue')
plt.title("Training data")
plt.xlabel("Years of EXP")
plt.ylabel("Salary")
plt.show()

