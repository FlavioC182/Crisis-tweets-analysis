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


def extraction_tagger(inputDataSet):

    list = []
    booleanList = []

    for count in range(0, len(inputDataSet.index)):
        list.append('None')
        booleanList.append(False)
    # adding 4 columns with default values
    inputDataSet["Location"] = pd.Series(list)
    inputDataSet["StateProvince"] = pd.Series(list)
    inputDataSet["Country"] = pd.Series(list)
    inputDataSet["City"] = pd.Series(list)

    inputDataSet["hasLocation"] = pd.Series(booleanList)
    inputDataSet["hasStateProvince"] = pd.Series(booleanList)
    inputDataSet["hasCountry"] = pd.Series(booleanList)
    inputDataSet["hasCity"] = pd.Series(booleanList)
    inputDataSet["hasCity"] = pd.Series(booleanList)

    # other tags:
    inputDataSet["hasTime"] = pd.Series(booleanList)
    inputDataSet["hasCriminal"] = pd.Series(booleanList)
    inputDataSet["hasDuration"] = pd.Series(booleanList)
    inputDataSet["hasDate"] = pd.Series(booleanList)
    inputDataSet["hasOrganization"] = pd.Series(booleanList)
    inputDataSet["hasDeath"] = pd.Series(booleanList)
    inputDataSet["hasMoney"] = pd.Series(booleanList)
    inputDataSet["hasIdeology"] = pd.Series(booleanList)
    inputDataSet["hasNumber"] = pd.Series(booleanList)
    inputDataSet["hasReligion"] = pd.Series(booleanList)
    inputDataSet["hasPerson"] = pd.Series(booleanList)
    inputDataSet["hasSet"] = pd.Series(booleanList)

    # adding tweet_id as index in order to be able to access to single values (cells) of dataframes with loc[id,column]
    # (using index and column_names)
    # probably it would have been easier using iloc, with a counter and without setting the index
    inputDataSet = inputDataSet.set_index('TweetID')
    # inputDataSet['tweet_id'] = inputDataSet.index

    # filtering text erasing hashtags:
    noHashDataSet = inputDataSet.copy()
    for id, tweet in zip(noHashDataSet.index.values, noHashDataSet["Text"]):
        try:
            noHashDataSet.loc[id, "Text"] = tweet.replace("#", "")
        except AttributeError:
            print(tweet)
            print(type(tweet))
            print(id)

    for id, tweet in zip(noHashDataSet.index.values, noHashDataSet["Text"]):
        Tagger = CoreNLPNERTagger(url='http://localhost:9000/')
        list = Tagger.tag(tweet.split())

        # in order to read the tags from i file
        # with open("Tags.txt",'a') as file:
        # file.write(str(list)+"\n")

        for tuple in list:
            if(tuple[1] == "LOCATION"):
                #inputDataSet.loc[id, "Location"] = tuple[0].lower()
                inputDataSet.loc[id, "hasLocation"] = True
            elif(tuple[1] == "STATE_OR_PROVINCE"):
                #inputDataSet.loc[id, "StateProvince"] = tuple[0].lower()
                inputDataSet.loc[id, "hasStateProvince"] = True
            elif(tuple[1] == "COUNTRY"):
                #inputDataSet.loc[id, "Country"] = tuple[0].lower()
                inputDataSet.loc[id, "hasCountry"] = True
            elif(tuple[1] == "CITY"):
                #inputDataSet.loc[id, "City"] = tuple[0].lower()
                inputDataSet.loc[id, "hasCity"] = True
            elif(tuple[1] == "TIME"):
                inputDataSet.loc[id, "hasTime"] = True
            elif(tuple[1] == "CRIMINAL_CHARGE"):
                inputDataSet.loc[id, "hasCriminal"] = True
            elif(tuple[1] == "DURATION"):
                inputDataSet.loc[id, "hasDuration"] = True
            elif(tuple[1] == "DATE"):
                inputDataSet.loc[id, "hasDate"] = True
            elif(tuple[1] == "ORGANIZATION"):
                inputDataSet.loc[id, "hasOrganization"] = True
            elif(tuple[1] == "CAUSE_OF_DEATH"):
                inputDataSet.loc[id, "hasDeath"] = True
            elif(tuple[1] == "MONEY"):
                inputDataSet.loc[id, "hasMoney"] = True
            elif(tuple[1] == "IDEOLOGY"):
                inputDataSet.loc[id, "hasIdeology"] = True
            elif(tuple[1] == "NUMBER"):
                inputDataSet.loc[id, "hasNumber"] = True
            elif(tuple[1] == "RELIGION"):
                inputDataSet.loc[id, "hasReligion"] = True
            elif(tuple[1] == "PERSON"):
                inputDataSet.loc[id, "hasPerson"] = True
            elif(tuple[1] == "SET"):
                inputDataSet.loc[id, "hasSet"] = True
    return inputDataSet


# Main method (to use only when this script is launched)
if __name__ == '__main__':
    inputDataSet = pd.read_csv(
        'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaDataFinal/2013_alberta_floods_metadati.csv', header=0)
    # erasing useless attributes and changing names
    inputDataSet = extraction_tagger(inputDataSet)
    inputDataSet.to_csv(r'MetaDataFinal/2013_alberta_floods_metadati_Flags.csv',
                        header=inputDataSet.columns.values, index=True,  sep=',', mode='w')
