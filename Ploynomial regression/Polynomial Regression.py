#ploynominal Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#import data...
os.getcwd()
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Fit Linear Regression
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X,y)

#set the degree for the ploynomial features degree decides how well the curve fits
from sklearn.preprocessing import PolynomialFeatures
pl=PolynomialFeatures(degree=4)
X_poly=pl.fit_transform(X)
# fit the new polynomial data to the linear modal
plm=LinearRegression()
plm.fit(X_poly,y)

#Visualize the Linear regression
plt.scatter(X,y,color='red')
plt.plot(X,lm.predict(X),color='blue')
plt.title('Linear Model')
plt.show()

#Visulaize the polynomial regression
plt.scatter(X,y,color='red')
plt.plot(X,plm.predict(pl.fit_transform(X)),color='blue')
plt.title('polynomial model')
plt.show()

#lets predict using linear model.
lm.predict(6.5)

#lets predict using polynomial modal.
plm.predict(pl.fit_transform(6.5))