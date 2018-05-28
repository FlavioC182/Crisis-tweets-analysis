import pandas as pd
import datetime as dt

#str_to_datetime: convert string data to number

months = {"Jan": 1,  "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7,
          "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

def str_to_datetime(stringTime):
    splittedDate = stringTime.split(" ")
    #splittedDate[1]: name of the month
    #splittedDate[2]: number of the month
    #splittedDate[3]: hour
    #splittedDate[5]: year
    splittedTime = splittedDate[3].split(":")
    dateNum = dt.datetime(int(splittedDate[5]), int(months[splittedDate[1]]), int(splittedDate[2]),
                          int(splittedTime[0]),int(splittedTime[1]),int(splittedTime[2]))
    return dateNum

# to calculate number of days from account creation
def usr_age(stringTime):
    splittedDate = stringTime.split(" ")
    #splittedDate[1]: name of the month
    #splittedDate[2]: number of the month
    #splittedDate[3]: hour
    #splittedDate[5]: year
    splittedTime = splittedDate[3].split(":")
    dateNum = dt.datetime(int(splittedDate[5]), int(months[splittedDate[1]]), int(splittedDate[2]),
                          int(splittedTime[0]),int(splittedTime[1]),int(splittedTime[2]))
    deltaD = dt.datetime.now() - dateNum
    # Extracting days from deltatime
    # Structure applying str(): xxxx Days, ...
    # This way str(now-registration_date).split(" ")[0]:
    #  - First splits the string using spaces as delimiter
    #  - Then returns the first element of the list
    return str(deltaD).split(" ")[0]

#to convert DeltaSeconds in integer:
def deltaSecToInt(stringTime):
    time = stringTime.split(":")
    return int(time[0])*60*60 + int(time[1])*60 + int(time[2])

#to erase useless attributes of tweets:
def selected_Attributes(dataset):
    # Deleting useless attributes
    del dataset["_unit_id"], dataset["_golden"], dataset["_unit_state"], dataset["_trusted_judgments"], dataset[
    "_last_judgment_at"], dataset["choose_one_category:confidence"], dataset["choose_one_category_gold"]

    #change attributes names:
    dataset.columns = ["Label","TweetID","Text"]
    dataset["TweetID"] = dataset["TweetID"].apply(remove_apexes)
    return dataset

#to use with apply of selected_Attributes
def remove_apexes(x):
    x = x.replace("'","")
    return x

#to join MetaData dataframes with metadata which has flags
def joinBetweenMetaDataAndCities(dataframe1,dataframe2):
    #they have to be the same index (TweetID)
    del dataframe2["Label"]
    dataframe1 = dataframe1.set_index("TweetID")
    dataframe2 = dataframe2.set_index("TweetID")
    joinDataFrame = dataframe1.join(dataframe2, how='inner')
    return joinDataFrame
