#function used to extract only irrelevant tweets (they are left in the base form, other preprocessing could be done)
import pandas as pd
import json
import numpy as np
from Utilities import selected_Attributes

inputDataSet = pd.read_csv('/Users/Flavio/Desktop/Tesi/Progetto/Dataset/2013_Pakistan_eq/2013_pakistan_eq.csv',header=0)

("caution_and_advice", "displaced_people_and_evacuations", "infrastructure_and_utilities_damage",
                   "injured_or_dead_people", "missing_trapped_or_found_people", "other_useful_information")

def irrelevant_Extraction(inputDataSet):
    inputDataSet = inputDataSet.drop(
    inputDataSet[ (inputDataSet["choose_one_category"] == "caution_and_advice") |
                  (inputDataSet["choose_one_category"] == "displaced_people_and_evacuations") |
                  (inputDataSet["choose_one_category"] == "infrastructure_and_utilities_damage") |
                  (inputDataSet["choose_one_category"] == "injured_or_dead_people") |
                  (inputDataSet["choose_one_category"] == "missing_trapped_or_found_people") |
                  (inputDataSet["choose_one_category"] == "other_useful_information")
                 ].index
    )

    return inputDataSet


irrelevantDataset = irrelevant_Extraction(inputDataSet)
irrelevantDataset = irrelevantDataset.set_index('tweet_id')
irrelevantDataset.to_csv(r'MetaData/2013_pakistan_irrilevant_text.csv', header=irrelevantDataset.columns.values, index=True,  sep=',', mode='w')
