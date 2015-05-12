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
## Program uses Statsmodels to run an OLS on the complete model
##########################################################################

if __name__== "__main__":
	#Import the data from excel as a dataframe
	df = pd.read_excel(DATAPATH,index_col=0)
	
	#Run the OLS regression model
	lm = smf.ols(formula='y ~ DRIVE + WHITE + LIQUOR_N + METRO_N + SINGLE + BA \
	+ CAMPUS_N + STARB_N + DENSITY + AGE + LANDMARK_N + LANDMARK_D + N_PARK_N + MCDON_D \
	+ STARB_D + WALK + BUS_N + POP + RENT + SINGLE + RENTER + TRANSIT + DC_PARK_D \
	+ LIQUOR_D + METRO_D + BUS_D + CAMPUS_D + MCDON_N + DC_PARK_N + CAMPUS_N', data=df).fit()
	
	#Print out the results
	Stats = lm.summary()
	print Stats
	
	
	
	
	
	
