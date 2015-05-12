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

##########################################################################
## Module Constants
##########################################################################
DIRNAME = os.path.dirname(__file__)
DATAPATH = os.path.join(DIRNAME, 'CapBikeDataLogUpdate.xlsx')
OUTPATH = os.path.join(DIRNAME, 'CapBikeCross.xls')


##########################################################################
## Program calculates the correlations of all the variables and outputs to excel file.
##########################################################################
if __name__== "__main__":
	#Import the data frame
	df = pd.read_excel(DATAPATH,index_col=0)
	#print df.head()
	
	#Get the labels for rows and columns
	Var_List = list(df.columns.values)[0:]

	#Create a numpy array from the dataframe
	Arr = np.array(df)
	#print Arr.shape
	
	#Transpose the numpy array for correlation coefficient computation
	Arr = np.transpose(Arr)
	#print Arr.shape
	
	#Compute the correlation coefficients
	Corr = np.corrcoef(Arr)
	
	#Create a dataframe of the array of correlation coefficients
	new_df = pd.DataFrame(Corr, index=Var_List, columns=Var_List )
	
	#Send the dataframe of correlation coefficients to an excel file
	new_df.to_excel(OUTPATH)
	
	
	#print new_df.head()
	
	
