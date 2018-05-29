#This code snippet is used to collect the usage of all the functions defined in other scripts, in order to simplify
#the creation of metadata, cleaning of tweets, and so on.

import pandas as pd
import numpy as np
from MetaDataAddingToTweets_wDF import metaDataExtraction
from ExtractionTagger import extraction_tagger
from Utilities import selected_Attributes
from Irrelevant_Tweets import irrelevant_Extraction

inputPakistan = '/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2013_Pakistan_eq/2013_pakistan_eq.csv'
inputCalifornia = '/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2014_California_Earthquake/2014_california_eq.csv'
inputPhilippines = '/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2014_Philippines_Typhoon_Hagupit_en/2014_typhoon_hagupit_cf_labels.csv'
inputChile = '/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2014_Chile_Earthquake_en/2014_chile_eq_en.csv'
inputIndia = '/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2014_India_floods/2014_india_floods.csv'
inputMexico = '/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2014_Hurricane_Odile_Mexico_en/2014_hurricane_odile.csv'

csvRead = inputChile

inputDataSet = pd.read_csv(csvRead, header=0)

#------------------------------------------------------------------------------------
# MetaData Extraction: comment to avoid usage

metaDataTweets = metaDataExtraction(inputDataSet)
col_names = list(metaDataTweets.columns.values)
metaDataTweets.to_csv(r'MetaData/2014_chile_metadati_WDF.csv', header=col_names, index=True, sep=',',mode='w')


#------------------------------------------------------------------------------------
# Cities Extraction: comment to avoid usage

cleanedDataset = extraction_tagger(inputDataSet)
cleanedDataset.to_csv(r'MetaData/2014_india_text_NLP.csv', header=cleanedDataset.columns.values, index=True,  sep=',', mode='w')


#------------------------------------------------------------------------------------
# Irrelevant tweet: comment to avoid usage

# to apply to metadata in order to filter irrelevant tweets
# - this one is not possible: metadata extraction + cities extractions

irrelevantMetaData = irrelevant_Extraction(metaDataTweets)
col_names = list(irrelevantMetaData.columns.values)
irrelevantMetaData.to_csv(r'MetaData/2014_chile_irrelevant_metadati.csv', header=col_names, index=True, sep=',',mode='w')

#------------------------------------------------------------------------------------
#To join NLP flags + MetaData or irrelevantMetaData
JoinDataFrame = joinBetweenMetaDataAndCities(metaDataTweets,cleanedDataset)
JoinDataFrame.to_csv(r'MetaData/2013_california_irrelevant_Metadata_source_Flags.csv', header=JoinDataFrame.columns.values, index=True, sep=',',mode='w')
