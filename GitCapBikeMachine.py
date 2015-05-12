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
## Allows for Model 1 and 2 features for regression results.
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
	
#Initialize the list to keep the scores from each iteration.
	OLS_score = []
	Ridge_score = []
	RidgeCV_score = []
	DecTree1_score = []
	DecTree2_score = []
	Lasso_score = []
	LassoCV_score = []
	RandomForest_score = []
	
		
# Obtain results for running the model a specified number of times
	for i in range(1,15):
#Train the data
		splits = tts(data, target, test_size=0.20)
		X_train, X_test, y_train, y_test = splits
	
#Run the OLS model.
		regr = linear_model.LinearRegression()
		regr.fit(X_train, y_train)
		OLS_score.append(regr.score(X_test, y_test))
		#print 'Coefficients OLS: \n', regr.coef_
		#print 'Intercept OLS: \n', regr.intercept_
		
#Run the Ridge model.
		clf = linear_model.Ridge(alpha=0.5)
		clf.fit(X_train, y_train)
		Ridge_score.append(clf.score(X_test, y_test))
		
#Run the RidgeCV model.
		n_alphas = 200
		alphas = np.logspace(-10,-2,n_alphas)
		clf = linear_model.RidgeCV(alphas=alphas, cv=12)
		clf.fit(X_train, y_train)
		RidgeCV_score.append(clf.score(X_test, y_test))
		#print 'Coefficients RidgeCV: \n', clf.coef_
		#print 'Intercept RidgeCV: \n', clf.intercept_		
		
#Run the Decision Tree model.
		clf_1 = DecisionTreeRegressor(max_depth=2)
		clf_2 = DecisionTreeRegressor(max_depth=5)
		clf_1.fit(X_train, y_train)
		clf_2.fit(X_train, y_train)
		DecTree1_score.append(clf_1.score(X_test, y_test))
		DecTree2_score.append(clf_2.score(X_test, y_test))
	
#Run the Lasso model.
		clf = linear_model.Lasso(alpha=0.5)
		clf.fit(X_train, y_train)
		Lasso_score.append(clf.score(X_test, y_test))
		
#Run the LassoCV model.
		n_alphas = 200
		alphas = np.logspace(-10,-2,n_alphas)
		clf = linear_model.LassoCV(alphas=alphas, cv=12)
		clf.fit(X_train, y_train)
		LassoCV_score.append(clf.score(X_test, y_test))
		
	
#Run the Random Forest model.
		clf = RandomForestRegressor()
		clf.fit(X_train, y_train)
		RandomForest_score.append(clf.score(X_test, y_test))
		#print 'Feature Importances Random Forest: \n', clf.feature_importances_
		
	
#Average and print results
	print
	
	#print "OLS", OLS_score
	OLS_score = filter(lambda d: d > 0, OLS_score)
	OLS_score_Avg = sum(OLS_score) / len(OLS_score)
	print "OLS Avg", OLS_score_Avg
	print
	

	#print "Ridge", Ridge_score
	Ridge_score = filter(lambda d: d > 0, Ridge_score)
	Ridge_score_Avg = sum(Ridge_score)/len(Ridge_score)
	print "Ridge Avg", Ridge_score_Avg
	print
	
	#print "RidgeCV", RidgeCV_score
	RidgeCV_score = filter(lambda d: d > 0, RidgeCV_score)
	RidgeCV_score_Avg = sum(RidgeCV_score)/len(RidgeCV_score)
	print "RidgeCV Avg", RidgeCV_score_Avg
	print
	
	#print "Decision Tree", DecTree1_score, DecTree2_score
	DecTree1_score = filter(lambda d: d > 0, DecTree1_score)
	DecTree2_score = filter(lambda d: d > 0, DecTree2_score)
	DecTree1_score_Avg = sum(DecTree1_score)/len(DecTree1_score)
	DecTree2_score_Avg = sum(DecTree2_score)/len(DecTree2_score)
	print "DecTree1 Avg", DecTree1_score_Avg
	print
	print "DecTree2 Avg", DecTree2_score_Avg
	print


	#print "Lasso", Lasso_score
	Lasso_score = filter(lambda d: d > 0, Lasso_score)
	Lasso_score_Avg = sum(Lasso_score)/len(Lasso_score)
	print "Lasso Avg", Lasso_score_Avg
	print
	
	#print "LassoCV", LassoCV_score
	LassoCV_score = filter(lambda d: d > 0, LassoCV_score)
	LassoCV_score_Avg = sum(LassoCV_score)/len(LassoCV_score)
	print "LassoCV Avg", LassoCV_score_Avg
	print
	
	
	#print "RandomForest", RandomForest_score
	RandomForest_score = filter(lambda d: d > 0, RandomForest_score)
	RandomForest_score_Avg = sum(RandomForest_score)/len(RandomForest_score)
	print "Random Forest Avg", RandomForest_score_Avg
	print
	
	
	
	
	