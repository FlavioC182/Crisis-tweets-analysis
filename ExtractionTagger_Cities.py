# Here we are going to use the Stanford NER Tagger in order to extract features related to a place from Text
# Then, columns with True/False flags are added in order to signal if in the Text there are informations related to place

# To make this code work it is important to download stanford-corenlp-full-2018-02-27
# check the dependencies (CLASSPATH and STANFORD_MODELS) and execute this command on cmd:
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# The above command has to be executed after having put yourself inside the corenlp-full-... folder
# A batch file (runServer.bat) for Windows is included in the repository, to
# easily execute the server. (It has to be placed in corenlp-full-... folder)

import pandas as pd
import json
import numpy as np
from Utilities import selected_Attributes
from nltk.tag.stanford import CoreNLPNERTagger

inputDataSet = pd.read_csv('/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2014_Philippines_Typhoon_Hagupit_en/2014_typhoon_hagupit_cf_labels.csv',header=0)

#erasing useless attributes and changing names
inputDataSet = selected_Attributes(inputDataSet)

def extraction_tagger(inputDataSet):
    list = []
    booleanList = []

    for count in range(0, len(inputDataSet.index)):
        list.append('None')
        booleanList.append(False)
    #adding 4 columns with default values
    inputDataSet["Location"] = pd.Series(list)
    inputDataSet["StateProvince"] = pd.Series(list)
    inputDataSet["Country"] = pd.Series(list)
    inputDataSet["City"] = pd.Series(list)

    inputDataSet["hasLocation"] = pd.Series(booleanList)
    inputDataSet["hasStateProvince"] = pd.Series(booleanList)
    inputDataSet["hasCountry"] = pd.Series(booleanList)
    inputDataSet["hasCity"] = pd.Series(booleanList)

    # adding tweet_id as index in order to be able to access to single values (cells) of dataframes with loc[id,column]
    # (using index and column_names)
    # probably it would have been easier using iloc, with a counter and without setting the index
    inputDataSet = inputDataSet.set_index('TweetID')
    # inputDataSet['tweet_id'] = inputDataSet.index

    #filtering text erasing hashtags:
    for id,tweet in zip(inputDataSet.index.values,inputDataSet["Text"]):
        inputDataSet.loc[id,"Text"] = tweet.replace("#","")

    for id, tweet in zip(inputDataSet.index.values,inputDataSet["Text"]):
        Tagger = CoreNLPNERTagger(url='http://localhost:9000/')
        list = Tagger.tag(tweet.split())
        for tuple in list:
            if(tuple[1]=="LOCATION"):
                inputDataSet.loc[id,"Location"] = tuple[0].lower()
                inputDataSet.loc[id,"hasLocation"] = True
            elif(tuple[1]=="STATE_OR_PROVINCE"):
                inputDataSet.loc[id,"StateProvince"] = tuple[0].lower()
                inputDataSet.loc[id,"hasStateProvince"] = True
            elif(tuple[1]=="COUNTRY"):
                inputDataSet.loc[id,"Country"] = tuple[0].lower()
                inputDataSet.loc[id,"hasCountry"] = True
            elif(tuple[1]== "CITY"):
                inputDataSet.loc[id,"City"] = tuple[0].lower()
                inputDataSet.loc[id,"hasCity"] = True
    return inputDataSet

inputDataSet = extraction_tagger(inputDataSet)
inputDataSet.to_csv(r'MetaData/2014_typhon_hagupit_text_NLP.csv', header=inputDataSet.columns.values, index=True,  sep=',', mode='w')
