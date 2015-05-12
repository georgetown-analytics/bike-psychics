# Author: Selma Gomez Orr <selmagomezorr@gmail.com> Copyright (C) May 2, 2015

##########################################################################
## Imports
##########################################################################

import os
import pandas as pd
import numpy as np
import xlrd
from sklearn.cross_validation import train_test_split as tts
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt 
from pandas.tools.plotting import scatter_matrix

from sklearn.feature_selection import f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

##########################################################################
## Module Constants
##########################################################################

DIRNAME = os.path.dirname(__file__)
DATAPATH = os.path.join(DIRNAME, 'CapBikeDataLogUpdate.xlsx')
GRAPHPATH = os.path.join(DIRNAME, 'CapBikeScatter.png')

##########################################################################
## Program runs a variety of regression types and plots y-predict vs. actual.
##########################################################################
#This program tests a variety of regression models on two subsets of variables.
if __name__== "__main__":
	#Import the data frames
	df = pd.read_excel(DATAPATH,index_col=0)
	
	print
	
#Determine the names of the dependent variables for Model 1
	#X_var = ['DRIVE', 'WHITE', 'LIQUOR_N', 'SINGLE', 'CAMPUS_N', 'BUS_N', 'MCDON_N']
	#print "Model 1 Results"
	
#Determine the names of the dependent variables for Model 2
	X_var = ['DRIVE', 'WHITE', 'DENSITY', 'AGE', 'WALK', 'BUS_N', 'TRANSIT', 'MCDON_N']
	print "Model 2 Results"
	
	#Convert the dataframe into numpy arrays for use with sklearn
	data = np.array(df[X_var])
	target = np.array(df['y'])
	

		
	
#Run the OLS model.
	regr = linear_model.LinearRegression()
	regr.fit(data, target)
	ypred = regr.predict(data)
	
#Plot the results of the actual versus the predicted value of y.	
	plt.scatter(ypred, target)
	plt.ylabel("Actual Y")
	plt.xlabel("Predicted Y")
	plt.title("OLS Actual Y versus Predicted Y")
	
	plt.show()

		
		
#Run the Ridge model.
	clf = linear_model.Ridge(alpha=0.5)
	clf.fit(data, target)
	ypred = clf.predict(data)

#Plot the results of the actual versus the predicted value of y.		
	plt.scatter(ypred, target)
	plt.ylabel("Actual Y")
	plt.xlabel("Predicted Y")
	plt.title("Ridge Actual Y versus Predicted Y")
	
	plt.show()
		
		
#Run the RidgeCV model.
	n_alphas = 200
	alphas = np.logspace(-10,-2,n_alphas)
	clf = linear_model.RidgeCV(alphas=alphas, cv=12)
	clf.fit(data, target)
	ypred = clf.predict(data)
	
#Plot the results of the actual versus the predicted value of y.		
	plt.scatter(ypred, target)
	plt.ylabel("Actual Y")
	plt.xlabel("Predicted Y")
	plt.title("RidgeCV Actual Y versus Predicted Y")
	
	plt.show()
		

		
#Run the Decision Tree model.
	clf_1 = DecisionTreeRegressor(max_depth=2)
	clf_2 = DecisionTreeRegressor(max_depth=5)
	clf_1.fit(data, target)
	clf_2.fit(data, target)
	
	
#Run the Lasso model.
	clf = linear_model.Lasso(alpha=0.5)
	clf.fit(data, target)
		
#Run the LassoCV model.
	n_alphas = 200
	alphas = np.logspace(-10,-2,n_alphas)
	clf = linear_model.LassoCV(alphas=alphas, cv=12)
	clf.fit(data, target)
	ypred = clf.predict(data)
	
#Plot the results of the actual versus the predicted value of y.		
	plt.scatter(ypred, target)
	plt.ylabel("Actual Y")
	plt.xlabel("Predicted Y")
	plt.title("LassoCV Actual Y versus Predicted Y")
	
	plt.show()
		
	
#Run the Random Forest model.
	clf = RandomForestRegressor()
	clf.fit(data, target)
	ypred = clf.predict(data)
	
#Plot the results of the actual versus the predicted value of y.	
	plt.scatter(ypred, target)
	plt.ylabel("Actual Y")
	plt.xlabel("Predicted Y")
	plt.title("Random Forest Actual Y versus Predicted Y")
	
	plt.show()
	
	
		
	

	
	
	