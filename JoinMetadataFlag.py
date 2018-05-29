# This code snippet is used to join csv (dataframes) created using the other codes.
# The join is between MetaData dataframe and Flags + text dataframe

import pandas as pd
import numpy as np
from Utilities import joinBetweenMetaDataAndCities
PakistanMetaData = '/Users/Flavio/Desktop/Tesi/Code/MetaDataExtraction/MetaData/2013_pakistan_metadati_WDF.csv'
PakistanIrrelevevantMD = '/Users/Flavio/Desktop/Tesi/Code/MetaDataExtraction/MetaData/2013_pakistan_irrelevant_metadati.csv'
PakistanFlags = '/Users/Flavio/Desktop/Tesi/Code/MetaDataExtraction/MetaData/2013_pakistan_Flags_NLP.csv'
CaliforniaMetaData = '/Users/Flavio/Desktop/Tesi/Code/MetaDataExtraction/MetaData/2014_california_metadati_source.csv'
CaliforniaFlags = '/Users/Flavio/Desktop/Tesi/Code/MetaDataExtraction/MetaData/2014_california_Flags_NLP.csv'
CaliforniaIrrelevantMD = '/Users/Flavio/Desktop/Tesi/Code/MetaDataExtraction/MetaData/2014_california_irrelevant_source_metadati.csv'

MetaDataF = pd.read_csv(CaliforniaIrrelevantMD, header=0)
FlagsDataF = pd.read_csv(CaliforniaFlags, header=0)

JoinDataFrame = joinBetweenMetaDataAndCities(MetaDataF,FlagsDataF)
JoinDataFrame.to_csv(r'MetaData/2013_california_irrelevant_Metadata_source_Flags.csv', header=JoinDataFrame.columns.values, index=True, sep=',',mode='w')
