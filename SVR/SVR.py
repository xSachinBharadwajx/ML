#SVR 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#import the dataset 
dataset =pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#feature Scaling is madatory in SVR because the library doesn't do it for you..
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y.reshape(-1,1))

#implement the 
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

#lets Visualize
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.show()

#lets predict
#but we need to inverse feature scale 
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(6.5)))