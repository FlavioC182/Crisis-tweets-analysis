# This code snippet is used to collect the usage of all the functions defined in other scripts, in order to simplify
# the creation of metadata, cleaning of tweets, and so on.

import pandas as pd
import numpy as np
from MetaDataAddingToTweets_wDF import metaDataExtraction
from ExtractionTagger import extraction_tagger
from Utilities import selected_Attributes
from Irrelevant_Tweets import irrelevant_Extraction
from Utilities import joinBetweenMetaDataAndCities

inputPakistan = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/SourceData/2013_pakistan_eq.csv'
inputCalifornia = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/SourceData/2014_california_eq.csv'
inputPhilippines = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/SourceData/2014_typhoon_hagupit_cf_labels.csv'
inputChile = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/SourceData/2014_chile_eq_en.csv'
inputIndia = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/SourceData/2014_india_floods.csv'
inputMexico = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/SourceData/2014_hurricane_odile.csv'
inputPakistan2014 = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/SourceData/2014_pakistan_floods_cf_labels.csv'
inputHagupit = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/SourceData/2014_typhoon_hagupit_cf_labels.csv'
csvRead = inputHagupit

inputDataSet = pd.read_csv(csvRead, header=0)

# ------------------------------------------------------------------------------------
# MetaData Extraction: comment to avoid usage

metaDataTweets = metaDataExtraction(inputDataSet)
col_names = list(metaDataTweets.columns.values)
metaDataTweets.to_csv(r'Metadata2/2014_hagupit_typhoon_metadati.csv',
                      header=col_names, index=True, sep=',', mode='w')


# ------------------------------------------------------------------------------------
# Cities Extraction: comment to avoid usage

#cleanedDataset = extraction_tagger(inputDataSet)
#cleanedDataset.to_csv(r'C:\dataset\2013_pakistan_metadati_complete.csv', header=cleanedDataset.columns.values, index=True,  sep=',', mode='w')


# ------------------------------------------------------------------------------------
# Irrelevant tweet: comment to avoid usage

# to apply to metadata in order to filter irrelevant tweets
# - this one is not possible: metadata extraction + cities extractions

irrelevantMetaData = irrelevant_Extraction(metaDataTweets)
col_names = list(irrelevantMetaData.columns.values)
irrelevantMetaData.to_csv(r'Metadata2/2014_hagupit_typhoon_irrelevant_metadati.csv',
                          header=col_names, index=True, sep=',', mode='w')

# ------------------------------------------------------------------------------------
# To join NLP flags + MetaData or irrelevantMetaData
cleanedDataset = pd.read_csv('https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_typhoon_hagupit_Flags_NLP.csv', header=0)
cleanedDataset = cleanedDataset.reset_index()
metaDataTweets = metaDataTweets.reset_index()
JoinDataFrame = joinBetweenMetaDataAndCities(metaDataTweets, cleanedDataset)
JoinDataFrame.to_csv(r'Metadata2/2014_hagupit_typhoon_metadati_flags.csv',
                     header=JoinDataFrame.columns.values, index=True, sep=',', mode='w')
