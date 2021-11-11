import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
all_tweets = pd.read_json("/Users/eddie/Desktop/twitter_classification_project/random_tweets.json", lines = True)

#print(all_tweets['user'].values)

all_tweets['is_viral'] = all_tweets['retweet_count'].apply(lambda retweets: 1 if retweets> 10000 else 0)

all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['is_verified'] = all_tweets['user'].apply(lambda status: 1 if status == True else 0)

all_tweets['hashtag_count'] = all_tweets.apply(lambda tweet: tweet['text'].count('#'), axis=1)
#print(all_tweets['hashtag_count'].mean())

x = all_tweets[['tweet_length', 'followers_count', 'is_verified', 'hashtag_count']]
y= all_tweets['is_viral']

scaled_data = scale(x, axis= 0)
print(scaled_data)

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size = 0.8, random_state= 42)

"""scores =[]
for n in range(1, 50):
    model = KNeighborsClassifier(n_neighbors = n)
    model.fit(train_x, train_y)
    score = model.score(test_x, test_y)
    scores.append(score)
plt.plot(scores)
plt.show()"""

model = KNeighborsClassifier(n_neighbors = 10)
model.fit(train_x, train_y)
score = model.score(test_x, test_y)
print(score)
eddie_data = [3, 200000000, False, 2]
prediction = model.predict([eddie_data])
print(prediction)
