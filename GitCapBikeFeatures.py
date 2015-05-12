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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.tree import DecisionTreeRegressor

##########################################################################
## Module Constants
##########################################################################
DIRNAME = os.path.dirname(__file__)
DATAPATH = os.path.join(DIRNAME, 'CapBikeDataLogUpdate.xlsx')
GRAPHPATH = os.path.join(DIRNAME, 'CapBikeScatter.png')


##########################################################################
## A program for exploration with features.  Not used for final project results.
##########################################################################
if __name__== "__main__":
	#Import the data frames
	df = pd.read_excel(DATAPATH,index_col=0)
	print df.head()


	#Get a summary of the data and print it.
	Stats = df.describe()
	print Stats
	
	#Create a scatter plot of all the variables and display it.
	scatter_matrix(df, alpha=0.2, figsize=(10,10), diagonal='kde')
	scatter_fig=plt.gcf()
	plt.show()
	
	#Save the scatter plot to file.
	scatter_fig.savefig(GRAPHPATH)
	
	#Get names of the dependent variables
	X_var = list(df.columns.values)[2:]
	
	
	#Convert the dataframe into numpy arrays for use with sklearn
	data = np.array(df[X_var])
	target = np.array(df['y'])
	print data.shape
	
	
	#Select K best features
	data_new = SelectKBest(f_regression, k=9).fit_transform(data, target)
	print data_new.shape
	

	#Train the data
	splits = tts(data_new, target, test_size=0.20)
	X_train, X_test, y_train, y_test = splits
	print X_train.shape
	print y_train.shape

	
	
	#Run the OLS model.
	regr = linear_model.LinearRegression()
	regr.fit(X_train, y_train)
	print
	print "OLS Results"
	print regr.coef_
	print regr.intercept_
	print mean_squared_error(y_test, regr.predict(X_test))
	print regr.score(X_test, y_test)
	
	
	#Run the Ridge model.
	clf = linear_model.Ridge(alpha=0.5)
	clf.fit(X_train, y_train)
	print 
	print "Ridge Results"
	print mean_squared_error(y_test, clf.predict(X_test))
	print clf.score(X_test, y_test)
	
	
	#Run the RidgeCV model.
	n_alphas = 200
	alphas = np.logspace(-10,-2,n_alphas)
	clf = linear_model.RidgeCV(alphas=alphas, cv=12)
	clf.fit(X_train, y_train)
	print
	print "CVRidge Results"
	print clf.alpha_
	print mean_squared_error(y_test, clf.predict(X_test))
	print clf.score(X_test, y_test)
	
	
	#Run the Decision Tree model.
	clf_1 = DecisionTreeRegressor(max_depth=2)
	clf_2 = DecisionTreeRegressor(max_depth=5)
	clf_1.fit(X_train, y_train)
	clf_2.fit(X_train, y_train)
	print
	print "Decision Tree Resutls"
	print mean_squared_error(y_test, clf_1.predict(X_test))
	print mean_squared_error(y_test, clf_2.predict(X_test))
	print clf_1.score(X_test, y_test)
	print clf_2.score(X_test, y_test)
	
	print data_new[2]
	
	
	
	
	
	
	
	