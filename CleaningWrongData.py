import pandas as pd
import numpy as np
import twython
from twython import Twython
import json
import time
import datetime as dt


pd.options.mode.chained_assignment = None  # default='warn'


def cleaning_DS(MetaDataFrame, BaseDataFrame):
    keys = json.load(open("keys.json"))

    CONSUMER_KEY = keys['CONSUMER_KEY']
    CONSUMER_SECRET = keys['CONSUMER_SECRET']
    OAUTH_TOKEN = keys['OAUTH_TOKEN']
    OAUTH_TOKEN_SECRET = keys['OAUTH_TOKEN_SECRET']

    twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

    MetaDataFrame = MetaDataFrame.set_index('TweetID')
    MetaDataFrame = MetaDataFrame[~MetaDataFrame.index.duplicated(keep='first')]
    MetaDataFrame = MetaDataFrame.reset_index(drop=False)

    BaseDataFrame = BaseDataFrame.set_index('tweet_id')
    BaseDataFrame = BaseDataFrame[~BaseDataFrame.index.duplicated(keep='first')]
    BaseDataFrame = BaseDataFrame.reset_index(drop=False)
    #in this way we can erase rows without problems during the for

    column_names = ['TweetID', 'CreationTime', 'Followers', 'Followed', 'GeoTagged', 'TotalTweets', 'TwitterAge',
                'nHashTags', 'nMentions', 'nUrls', 'Verified', 'Label', 'nRetweets', 'nLikes', 'Source', 'isRetweet']
    CleanedDataFrame = pd.DataFrame(columns=column_names)

    extractedTweets = 0

    for id, count in zip(MetaDataFrame['TweetID'],range(0, len(MetaDataFrame.index))):
        while True:
            try:
                cur_tweet = twitter.show_status(id=id)
                print(cur_tweet['text'])
                for indexT in BaseDataFrame.index.values:
                    if(cur_tweet['text'] == BaseDataFrame.loc[indexT,'tweet_text']):
                        #copy the row inside the new DT
                        CleanedDataFrame.loc[count] = MetaDataFrame.loc[count]
                        CleanedDataFrame.loc[count,'Label'] = BaseDataFrame.loc[indexT,'choose_one_category']
                        CleanedDataFrame.loc[count,'Text'] = BaseDataFrame.loc[indexT,'choose_one_category']
                        print("Find a Row")
                        with open("HagupitCleaned.txt",'a') as file:
                            file.write("Before substitution:   "+str(MetaDataFrame.loc[count,'TweetID'])+" "+str(MetaDataFrame.loc[count,'Label'])+" "+str(MetaDataFrame.loc[count,'Text'])+"\n")
                            file.write("After substitution:    "+str(CleanedDataFrame.loc[count,'TweetID'])+" "+str(CleanedDataFrame.loc[count,'Label'])+" "+str(CleanedDataFrame.loc[count,'Text'])+"\n")
                            extractedTweets = extractedTweets + 1
                print("No match")
                break
            except twython.exceptions.TwythonRateLimitError as e:
                # If we reach the limit of downloadable tweets in a time window, we wait 5 minutes and try again
                print(e)
                print("[DEBUG] Maximum number of tweets reached. Trying again in 5 mins...")
                print("[DEBUG]", extractedTweets, "tweets have been correctly keeped")
                for i in range(300):
                    if i == 299:
                        print("[DEBUG]", 300-i, "second left...")
                    else:
                        print("[DEBUG]", 300-i, "seconds left...")
                    time.sleep(1)
            except twython.exceptions.TwythonError as e:
                # If other exceptions occurs (APIs return an unexpected HTTP response code) we print it
                # This box includes error like 404 - Not found, 403 - User suspended etc.
                print("[DEBUG]", e)
                break
    CleanedDataFrame = CleanedDataFrame.set_index('TweetID')
    return CleanedDataFrame


if __name__ == '__main__':
    inputCaliforniaMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_california_metadati_Flags.csv'
    inputCaliforniaBase = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/SourceData/2014_california_eq.csv'
    inputPakistanMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2013_pakistan_metadati_Flags.csv'
    inputChileMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_chile_metadati_Flags.csv'
    inputOdileMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_odile_hurricane_metadati_Flags.csv'
    inputHagupitMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_hagupit_typhoon_metadati_flags.csv'

    MetaDataFrame = pd.read_csv(inputHagupitMD, header = 0)
    BaseDataFrame = pd.read_csv(inputCaliforniaBase, header = 0)
    print(MetaDataFrame.index.name)

    cleanedDS = cleaning_DS(MetaDataFrame, BaseDataFrame)
    cleanedDS.to_csv(r'MetaData3/2014_California_Cleaned_metadati_Flags_2.csv', header=cleanedDS.columns.values, index=True, sep=',', mode='w')
