# Author: Selma Gomez Orr <selmagomezorr@gmail.com> Copyright (C) May 2, 2015

##########################################################################
## Imports
##########################################################################
import os
import pandas as pd
import numpy as np
import xlrd


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor


##########################################################################
## Module Constants
##########################################################################
DIRNAME = os.path.dirname(__file__)
DATAPATH = os.path.join(DIRNAME, 'CapBikeDataLogUpdate.xlsx')


##########################################################################
## Program based on OLS for prediction and Model 2 for features.
##########################################################################
#Takes input from the user and predicts bike station popularity.
if __name__== "__main__":
	#Import the data frames
	df = pd.read_excel(DATAPATH,index_col=0)
	
	
#Determine the names of the dependent variables for Model 1
	X_var = ['DRIVE', 'WHITE', 'LIQUOR_N', 'SINGLE', 'CAMPUS_N', 'BUS_N', 'MCDON_N']

	
	
#Convert the dataframe into numpy arrays for use with sklearn
	data = np.array(df[X_var])
	target = np.array(df['y'])
	
	
#Run the OLS model.
	regr = linear_model.LinearRegression()
	regr.fit(data, target)


 	
	
	X_new = [0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0]
	Drive_data = input("Please provide the values for Drive percentage:")
	X_new[0] = Drive_data
	White_data = input("Please provide the values for White percentage:")
	X_new[1] = White_data
	Liquor_data = input("Please provide the values for nearby liquor licenses:")
	X_new[2] = Liquor_data
	Single_data = input("Please provide the values for Single percentage:")
	X_new[3] = Single_data
	Campus_data = input("Please provide the number of nearby campuses:")
	X_new[4] = Campus_data
	Bus_data = input("Please provide the number of bus stations nearby:")
	X_new[5] = Bus_data
	McDon_data = input("Please provide the number of nearby McDonald's:")
	X_new[6] = McDon_data
	
	print
	print "To confirm, these are your input values:", X_new	
	
	
#Use the user provided input to predict a value for proposed station popularity	
	ypred = regr.predict(X_new)

	
#Print the results.

	print
	print
	print "The projected popularity rating is", ypred
	if ypred > 2.5:
		print "This location has potential."
	else:
		print "You might want to find another location."
	print
	print
	
	
	

	

		
	

	
	
	