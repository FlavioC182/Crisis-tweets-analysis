import pandas as pd
import numpy as np
import twython
from twython import Twython
import json
import time
import datetime as dt


pd.options.mode.chained_assignment = None  # default='warn'


def cleaning_DS(BaseDataFrame,MetaDataFrame):
    keys = json.load(open("keys.json"))

    CONSUMER_KEY = keys['CONSUMER_KEY']
    CONSUMER_SECRET = keys['CONSUMER_SECRET']
    OAUTH_TOKEN = keys['OAUTH_TOKEN']
    OAUTH_TOKEN_SECRET = keys['OAUTH_TOKEN_SECRET']

    twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

    for count in range(0, len(BaseDataFrame.index)):
        BaseDataFrame.loc[count, 'tweet_id'] = BaseDataFrame.loc[count, 'tweet_id'].replace("'", "")

    BaseDataFrame = BaseDataFrame.set_index('tweet_id')
    #to eliminate duplicates
    BaseDataFrame = BaseDataFrame[~BaseDataFrame.index.duplicated(keep='first')]
    MetaDataFrame = MetaDataFrame.set_index('TweetID')

    #in this way we can erase rows without problems during the for
    CleanedDataFrame = MetaDataFrame.copy()
    extractedTweets = 0

    for id in MetaDataFrame.index.values:
        while True:
            try:
                cur_tweet = twitter.show_status(id=id)
                print(BaseDataFrame.loc[str(id),'tweet_text'])
                print(cur_tweet['text'])
                if(cur_tweet['text'] != BaseDataFrame.loc[str(id),'tweet_text']):
                    #remove a row (inplace allows to not reassign the dataframe)
                    CleanedDataFrame.drop(id,axis=0,inplace=True)
                    print("Erasing row")
                else:
                    print("Keeping row")
                    with open("CaliforniaCleaned.txt",'a') as file:
                        file.write("Twitter Obj: "+str(cur_tweet['user']['id'])+" "+str(cur_tweet['user']['name'])+" "+str(id)+" "+str(cur_tweet['text'])+"\n")
                        file.write("CSV: "+str(id)+" "+BaseDataFrame.loc[id,'tweet_text']+"\n")
                        extractedTweets = extractedTweets + 1
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
    return CleanedDataFrame


if __name__ == '__main__':
    inputCaliforniaBase = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/SourceData/2014_california_eq.csv'
    inputCaliforniaMD = 'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/MetaData2/2014_california_metadati_Flags.csv'
    BaseDataFrame = pd.read_csv(inputCaliforniaBase,header = 0)
    MetaDataFrame = pd.read_csv(inputCaliforniaMD, header = 0)
    print(BaseDataFrame.index.name)
    print(MetaDataFrame.index.name)
    cleanedDS = cleaning_DS(BaseDataFrame, MetaDataFrame)
    cleanedDS.to_csv(r'MetaData3/2014_california_Cleaned_metadati_Flags.csv', header=col_names, index=True, sep=',', mode='w')
