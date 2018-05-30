#function used to extract only irrelevant tweets (they are left in the base form, other preprocessing could be done)
import pandas as pd
import json
import numpy as np
from Utilities import selected_Attributes


def irrelevant_Extraction(inputDataSet):
    inputDataSet = inputDataSet.drop(

    #Label or choose_one_category. It depends on input file

    inputDataSet[ (inputDataSet["Label"] == "caution_and_advice") |
                  (inputDataSet["Label"] == "displaced_people_and_evacuations") |
                  (inputDataSet["Label"] == "infrastructure_and_utilities_damage") |
                  (inputDataSet["Label"] == "injured_or_dead_people") |
                  (inputDataSet["Label"] == "missing_trapped_or_found_people") |
                  (inputDataSet["Label"] == "other_useful_information")
                 ].index
    )
    #It is fundamental to re index indices (since there would be some "holes" in the index rows)
    #useless if index is TweetID
    #inputDataSet = inputDataSet.reset_index(drop = True)

    return inputDataSet

if __name__ == '__main__':
    inputDataSet = pd.read_csv('https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_california_metadati.csv',header=0)
    irrelevantDataset = irrelevant_Extraction(inputDataSet)
    irrelevantDataset = irrelevantDataset.set_index('TweetID')
    irrelevantDataset.to_csv(r'MetaData2/2014_california_irrelevant_metadati.csv', header=irrelevantDataset.columns.values, index=True,  sep=',', mode='w')
