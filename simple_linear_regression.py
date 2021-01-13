# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd 
import pylab as pl
import numpy as np
%matplotlib inline

import wget 
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
file_name=wget.filename_from_url(url)
print(file_name)

df=pd.read_csv(url)
#take a look at the dataset
df.head()

#summarize the data
df.describe()
cdf=df[['ENGINESIZE','CYLINDERS','FUELCOMSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
#select some features to plot
viz=cdf[['ENGINESIZE','CYLINDERS','FUELCOMSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()

# scatter plot_1
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
# scatter plot_2
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
# scatter_plot_3
plt.scatter(cdf.CYLINDER, cdf.CO2EMISSIONS, color='blue')
plt.xlabel=("CYLINDER")
plt.ylabel=("Emission")
plt.show()

# creating test dataset, 80%of the entire data for training, and the 20% for testing
# we create a mask to select from random row
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

# modeling
from sklearn import linear_model
regr=linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x,train_y)
print('coefficients:',regr.coef_)
print('intercept:', regr.intercept_)

# plot outputs: we can plot the fit line over the datas
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,color='blue')
plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0],'-r')
plt.xlabel("Engine Size")
plt.ylabel("Emissions")

# evaluation: Mean absolute error, residual sum of square, r square metric
from sklearn.metrics import r2_score
test_x=np.asanyarray(test[['ENGINESIZE']])
test_y=np.asanyarray(test[['CO2EMISSIONS']])
test_y=regr.predict(test_x)
print("Mean Abusolute Error: %.2f" % np.mean(np.absolute(test_y-test_x)))
print("Residual sum of squares (MSE):%.2f" % np.mean(test_y-test_x)**2 )
print("R Square: %.2f" % r2_score(test_y, test_x))







