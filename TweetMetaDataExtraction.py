# This code snippet is in charge of extract metadatas from twitter by means of its API.
# As input, a csv with a tweet_id is used in order to find the correct tweet and to extract all the useful metadatas.
import twython
from twython import Twython
import pandas as pd
import json
import numpy as np
import time
import datetime as dt
from Utilities import usr_age
from Utilities import str_to_datetime
from Utilities import deltaSecToInt
from Utilities import extract_Source

pd.options.mode.chained_assignment = None  # default='warn'


# defined a function in order to exploit it in other scripts if it is useful
def metaDataExtraction(myDataset):
    keys = json.load(open("keys.json"))

    CONSUMER_KEY = keys['CONSUMER_KEY']
    CONSUMER_SECRET = keys['CONSUMER_SECRET']
    OAUTH_TOKEN = keys['OAUTH_TOKEN']
    OAUTH_TOKEN_SECRET = keys['OAUTH_TOKEN_SECRET']

    twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

    # ix is used to access to every single value of the column "tweet_id" in order to cancel '
    #len(myDataset.index) or myDataset["tweet_id"].size

    extractedTweets = 0
    not_available = 0
    counter = 0
    firstTweetTime = dt.datetime.max
    column_names = ['TweetID', 'CreationTime', 'Followers', 'Followed', 'GeoTagged', 'TotalTweets', 'TwitterAge',
                    'nHashTags', 'nMentions', 'nUrls', 'Verified', 'nRetweets', 'nLikes', 'Source', 'isRetweet', 'Text', 'InfoSource', 'InfoType','Informativeness']
    endDataset = pd.DataFrame(columns=column_names)
    print("il tipo di endDataset e'")
    print(type(endDataset))

    for id, infoSource, infoType, informativenes in zip(myDataset["Tweet ID"], myDataset[" Information Source"],myDataset[" Information Type"], myDataset[" Informativeness"]):
        while True:
            try:
                cur_tweet = twitter.show_status(id=id)
                endDataset.loc[counter] = [str(id), cur_tweet['created_at'], cur_tweet['user']['followers_count'],
                                          cur_tweet['user']['friends_count'], cur_tweet['user']['geo_enabled'],
                                          cur_tweet['user']['statuses_count'], usr_age(cur_tweet['user']['created_at']),
                                          len(cur_tweet['entities']['hashtags']), len(cur_tweet['entities']['user_mentions']),
                                          len(cur_tweet['entities']['urls']), cur_tweet['user']['verified'],
                                          cur_tweet['retweet_count'], cur_tweet['favorite_count'], extract_Source(str(cur_tweet['source'])),
                                          "retweeted_status" in cur_tweet, myDataset.loc[counter,' Tweet Text'] , infoSource, infoType, informativenes]

                #with open("Australia.txt",'a') as file:
                #    file.write(str(id)+"\n")
                #with open("AustraliaFloods.txt",'a') as file:
                #  file.write("Twitter Obj: "+str(id)+" "+str(cur_tweet['user']['id'])+" "+str(cur_tweet['user']['name'])+" "+str(cur_tweet['text'])+"\n")
                #  file.write("CSV: "+str(myDataset.loc[counter,'Tweet ID'])+" "+myDataset.loc[counter,' Tweet Text']+"\n")

                extractedTweets = extractedTweets + 1
                print("[DEBUG] Found info for tweet: ", id, ". Added to list.")
                # deltatime starts from the first tweet related to the event
                if (informativenes == "Related and informative" or informativenes == "Related - but not informative"):
                    if firstTweetTime > str_to_datetime(cur_tweet['created_at']):
                        firstTweetTime = str_to_datetime(cur_tweet['created_at'])
                break
            except twython.exceptions.TwythonRateLimitError as e:
                # If we reach the limit of downloadable tweets in a time window, we wait 5 minutes and try again
                print(e)
                print("[DEBUG] Maximum number of tweets reached. Trying again in 5 mins...")
                print("[DEBUG]", extractedTweets, "tweets have been correctly extracted")
                print("[DEBUG]", not_available,
                      "tweets have encountered problems during download (403, 404, ...)")
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
                not_available = not_available + 1
                break
        counter = counter + 1

    deltaSec = []
    for date in endDataset["CreationTime"]:
        deltaSec.append(deltaSecToInt(str(str_to_datetime(date)-firstTweetTime)))
    endDataset["DeltaSeconds"] = deltaSec
    # to avoid the presence of a column of integer indexes in the csv
    endDataset = endDataset.set_index('TweetID')
    return endDataset


# Main method (to use only when this script is launched)
if __name__ == '__main__':
    myDatasetInput = pd.read_csv(
        'https://raw.githubusercontent.com/FlavioC182/Crisis-tweets-analysis/master/CrisisLexT26/2013_West_Texas_explosion/2013_West_Texas_explosion-tweets_labeled.csv', header=0)
    endDataset = metaDataExtraction(myDatasetInput)
    col_names = list(endDataset.columns.values)
    endDataset.to_csv(r'MetaDataFinal/2013_west_texas_explosion_metadati.csv',
                      header=col_names, index=True, sep=',', mode='w')
