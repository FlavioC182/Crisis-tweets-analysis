#function used to extract only relevant tweets from metadata_flag files.


import pandas as pd
import json
import numpy as np
from Utilities import selected_Attributes


def irrelevant_Extraction(inputDataSet):

    relevant_labels = ("caution_and_advice", "displaced_people_and_evacuations", "infrastructure_and_utilities_damage",
                   "injured_or_dead_people", "missing_trapped_or_found_people")

    resultDataSet = inputDataSet[inputDataSet["Label"].isin(relevant_labels)]

    #It is fundamental to re index indices (since there would be some "holes" in the index rows)
    #useless if index is TweetID
    #inputDataSet = inputDataSet.reset_index(drop = True)

    return resultDataSet


if __name__ == '__main__':
    inputDataSet = pd.read_csv('https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2013_pakistan_metadati_Flags.csv',header=0)
    irrelevantDataset = irrelevant_Extraction(inputDataSet)
    irrelevantDataset = irrelevantDataset.set_index('TweetID')
    irrelevantDataset.to_csv(r'MetaData2/2014_california_relevant_metadati_Flags.csv', header=irrelevantDataset.columns.values, index=True,  sep=',', mode='w')
