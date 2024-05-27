
import snscrape.modules.twitter as sntwitter
import pandas as pd

df2 = pd.read_csv('list_companies.csv')
tweets = []
limit = 250000

for i in df2['Company Name']:
    query = f"{i} lang:en until:2023-06-20 since:2018-01-01"

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append([i, tweet.date, tweet.username, tweet.content, tweet.likeCount, tweet.replyCount, tweet.retweetCount])
        
        if len(tweets) == limit:
            break
    
    if len(tweets) == limit:
        break

df = pd.DataFrame(tweets, columns=['Company', 'Date', 'User_name', 'Tweet', 'Likes', 'Comments', 'Shares'])
df.to_csv('Twitter_Data.csv', index=False)
