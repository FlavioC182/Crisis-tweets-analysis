#DEPRECATED: SEE _WDF VERSION
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

pd.options.mode.chained_assignment = None #default='warn'

keys = json.load(open("keys.json"))

CONSUMER_KEY = keys['CONSUMER_KEY']
CONSUMER_SECRET = keys['CONSUMER_SECRET']
OAUTH_TOKEN = keys['OAUTH_TOKEN']
OAUTH_TOKEN_SECRET = keys['OAUTH_TOKEN_SECRET']

twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

myDataset = pd.read_csv('../../Progetto/Dataset/2014_California_Earthquake/2014_california_eq.csv',header=0)

#ix is used to access to every single value of the column "tweet_id" in order to cancel '
#len(myDataset.index) or myDataset["tweet_id"].size
for count in range(0, len(myDataset.index)):
    myDataset.loc[count,'tweet_id'] = myDataset.loc[count,'tweet_id'].replace("'","")

print("[DEBUG] Formatting tweets")
print(myDataset.loc[0:10,"tweet_id"])

endDataset = []
extractedTweets = 0
not_available = 0
firstTweetTime = dt.datetime.max
print("il tipo di endDataset e'")
print(type(endDataset))

for id, category in zip(myDataset["tweet_id"],myDataset["choose_one_category"]):
    while True:
        try:
            cur_tweet = twitter.show_status(id=id)

            endDataset.append({'TweetID' : id,'CreationTime' : cur_tweet['created_at'], 'Followers' : cur_tweet['user']['followers_count'],
                       'Followed' : cur_tweet['user']['friends_count'], 'GeoTagged' : cur_tweet['user']['geo_enabled'],
                       'TotalTweets' : cur_tweet['user']['statuses_count'], 'TwitterAge' : usr_age(cur_tweet['user']['created_at']),
                       'nHashTags' : len(cur_tweet['entities']['hashtags']), 'nMentions': len(cur_tweet['entities']['user_mentions']),
                       'nUrls': len(cur_tweet['entities']['urls']), 'Verified': cur_tweet['user']['verified'], 'Label' : category,
                       'nRetweets' : cur_tweet['retweet_count'], 'nLikes' : cur_tweet['favorite_count']})

            extractedTweets = extractedTweets + 1
            print("[DEBUG] Found info for tweet: ", id, ". Added to list.")
            if firstTweetTime > str_to_datetime(cur_tweet['created_at']):
                firstTweetTime = str_to_datetime(cur_tweet['created_at'])
            break
        except twython.exceptions.TwythonRateLimitError as e:
        # If we reach the limit of downloadable tweets in a time window, we wait 5 minutes and try again
            print(e)
            print("[DEBUG] Maximum number of tweets reached. Trying again in 5 mins...")
            print("[DEBUG]", extractedTweets, "tweets have been correctly extracted")
            print("[DEBUG]", not_available, "tweets have encountered problems during download (403, 404, ...)")
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

deltaSec = []
for date in endDataset:
    date["DeltaSeconds"] = deltaSecToInt(str(str_to_datetime(date["CreationTime"]) - firstTweetTime))


tableTweet = pd.DataFrame(endDataset)
tableTweet = tableTweet.set_index('TweetID')
col_names = list(tableTweet.columns.values)

tableTweet.to_csv(r'2014_california_metadati.csv', header=col_names, index=True, sep=',',mode='w')
