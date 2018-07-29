#Multiple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset and assign to X the independent variables and y the dependant variable 
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#Encode the data in the statecoloumn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lblencoder=LabelEncoder()
X[:,3] =lblencoder.fit_transform(X[:,3])
onehtecdr= OneHotEncoder(categorical_features=[3])
X=onehtecdr.fit_transform(X).toarray()


#Avoiding Dummy variable Trap by removing one of the coloumn
X=X[:,1:]

#split the data into training and test data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Implement Linar regression
from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(X_train,y_train)

#Predit with the test set 
y_pred=linear.predict(X_test)


#Remove coloumns that are not significant by using backward Elemination.
import statsmodels.formula.api as sm
#be need to add a couloum on ones to the input variables 
#because y=b0+b1*x1+....bn*xn we need to accomadate b0...
#axis sets the row or coloumn, if axis =1 it adds as coloumn...
X=np.append(arr=np.ones(shape=(50,1)).astype(int),values=X,axis=1)
#step 1: assign all the elements.
X_opt=X[:,[0,1,2,3,4,5]]
#step 2: get the current p value of each row 
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#step 3: get the highest value to p and remove it and repeat.
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
