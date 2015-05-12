# Author: Selma Gomez Orr <selmagomezorr@gmail.com> Copyright (C) May 2, 2015

##########################################################################
## Imports
##########################################################################
import os
import xlrd
import pandas as pd

DIRNAME = os.path.dirname(__file__)
DATAPATH = os.path.join(DIRNAME, "Starbucks.xlsx")
OUTPATH = os.path.join(DIRNAME, "Starbucks.csv")


#Make the necessary imports for GeoPy functionality
import geopy
from geopy import geocoders
from geopy.geocoders import GeocoderDotUS

##########################################################################
## Module Constants
##########################################################################

FEATURES = ["Address", "Latitude", "Longitude"]

##########################################################################
## Program converts addresses to latitude and longitude (Starbucks, McDonald's) to CSV file.
##########################################################################
if __name__== "__main__":
	#Import the data frame
	df = pd.read_excel(DATAPATH,header=None, names=FEATURES)
	
	
	#Loop through addresses for each coordinate
	geolocator = GeocoderDotUS(format_string="%s, Washington DC")
	for row in range (0,48):
		address, coordinates = geolocator.geocode(df["Address"][row])
		df.loc[row, "Address"] = address
		df.loc[row, "Latitude"] = coordinates[0]
		df.loc[row, "Longitude"] = coordinates[1]
		
	
	#Write the results to a CSV file
	df.to_csv(OUTPATH, index=False)
	
	
		
		
		
		







