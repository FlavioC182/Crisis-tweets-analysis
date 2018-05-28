# This code snippet is used to join csv (dataframes) created using the other codes.
# The join is between MetaData dataframe and Flags + text dataframe

import pandas as pd
import numpy as np
from Utilities import joinBetweenMetaDataAndCities
PakistanMetaData = '/Users/Flavio/Desktop/Tesi/Code/MetaDataExtraction/MetaData/2013_pakistan_metadati_WDF.csv'
PakistanFlags = '/Users/Flavio/Desktop/Tesi/Code/MetaDataExtraction/MetaData/2013_pakistan_Flags_NLP.csv'
CaliforniaMetaData = '/Users/Flavio/Desktop/Tesi/Code/MetaDataExtraction/MetaData/2014_california_metadati_WDF.csv'
CaliforniaFlags = '/Users/Flavio/Desktop/Tesi/Code/MetaDataExtraction/MetaData/2014_california_Flags_NLP.csv'

MetaDataF = pd.read_csv(CaliforniaMetaData, header=0)
FlagsDataF = pd.read_csv(CaliforniaFlags, header=0)

JoinDataFrame = joinBetweenMetaDataAndCities(MetaDataF,FlagsDataF)
JoinDataFrame.to_csv(r'MetaData/2013_pakistan_Metadata_Flags_NLP.csv', header=JoinDataFrame.columns.values, index=True, sep=',',mode='w')
