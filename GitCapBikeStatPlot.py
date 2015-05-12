# Author: Selma Gomez Orr <selmagomezorr@gmail.com> Copyright (C) May 2, 2015

##########################################################################
## Imports
##########################################################################
import os
import pandas as pd
import numpy as np

import xlrd
import xlwt

import matplotlib.pyplot as plt 
from pandas.tools.plotting import scatter_matrix

import statsmodels.formula.api as smf


##########################################################################
## Module Constants
##########################################################################
DIRNAME = os.path.dirname(__file__)
DATAPATH = os.path.join(DIRNAME, 'CapBikeDataLogUpdate.xlsx')


##########################################################################
## Program uses Statsmodles OLS to plot y-predicted versus actual.
##########################################################################
if __name__== "__main__":
#Import the data from excel as a dataframe
	df = pd.read_excel(DATAPATH,index_col=0)
	
#Run the OLS regression model
	lm = smf.ols(formula='y ~ DRIVE + WHITE + DENSITY + AGE + BUS_N + WALK \
	+ MCDON_N + TRANSIT', data=df).fit()
	
	X_var = ['DRIVE', 'WHITE', 'DENSITY', 'AGE', 'BUS_N', 'WALK', 'MCDON_N', 'TRANSIT']
	X = df[X_var]
	
#Obtain the value of y-predicted	
	yact = df['y']
	ypred = lm.predict(X)

#Plot the results of the actual versus the predicted value of y.	
	plt.scatter(ypred, yact)
	plt.ylabel("Actual Y")
	plt.xlabel("Predicted Y")
	plt.title("Actual Y versus Predicted Y")
	
	plt.show()
	
	
	
	
	
	
	
