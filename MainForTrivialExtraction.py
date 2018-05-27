#This code snippet is used to collect the usage of all the functions defined in other scripts, in order to simplify
#the creation of metadata, cleaning of tweets, and so on.

import pandas as pd
import numpy as np
from MetaDataAddingToTweets_wDF import metaDataExtraction
from ExtractionTagger_Cities import extraction_tagger
from Utilities import selected_Attributes
from Irrelevant_Tweets import irrelevant_Extraction

inputPakistan = '/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2013_Pakistan_eq/2013_pakistan_eq.csv'
inputCalifornia = '/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2014_California_Earthquake/2014_california_eq.csv'
inputPhilippines = '/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2014_Philippines_Typhoon_Hagupit_en/2014_typhoon_hagupit_cf_labels.csv'
inputChile = '/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2014_Chile_Earthquake_en/2014_chile_eq_en.csv'

inputDataSet = pd.read_csv(inputChile, header=0)

# MetaData Extraction: comment to avoid usage

metaDataTweets = metaDataExtraction(inputDataSet)

metaDataTweets.set_index('TweetID')
col_names = list(metaDataTweets.columns.values)

metaDataTweets.to_csv(r'MetaData/2014_chile_metadati_WDF.csv', header=col_names, index=True, sep=',',mode='w')


# Cities Extraction: comment to avoid usage

cleanedDataset = selected_Attributes(inputDataSet)
cleanedDataset = extraction_tagger(cleanedDataset)
cleanedDataSet.to_csv(r'MetaData/2014_chile_text_NLP.csv', header=cleanedDataSet.columns.values, index=True,  sep=',', mode='w')

# Irrelevant tweet: comment to avoid usage

# Warning: it is important to apply only one of the following after the irrelevant_Extraction
# A) metadata extraction
# B) cleaning attributes (selected_Attributes function) + cities extraction
# - this one is not possible: metadata extraction + cities extractions

irrilevantDataSet = irrelevant_Extraction(inputDataSet)

# A)
irrelevantMetaData = metaDataExtraction(irrelevantDataSet)
irrelevantMetaData.set_index('TweetID')
col_names = list(irrelevantMetaData.columns.values)
irrilevantMetaData.to_csv(r'MetaData/2014_chile_irrelevant_metadati.csv', header=col_names, index=True, sep=',',mode='w')

# B)
irrelevantCleaned = selected_Attributes(irrelevantDataSet)
citiesIrrilevant = extraction_tagger(irrelevantCleaned)
citiesIrrilevant.to_csv(r'MetaData/2014_chile_text_irrilevant_NLP.csv', header=citiesIrrilevant.columns.values, index=True,  sep=',', mode='w')
