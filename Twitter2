import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
new_york_tweets = pd.read_json("/Users/eddie/Desktop/twitter_classification_project_2/new_york.json", lines= True)
paris_tweets = pd.read_json("/Users/eddie/Desktop/twitter_classification_project_2/paris.json", lines = True)
london_tweets = pd.read_json("/Users/eddie/Desktop/twitter_classification_project_2/london.json", lines = True)

#print(new_york_tweets.columns)

#print(len(paris_tweets))

new_york_text = new_york_tweets['text'].tolist()
paris_text = paris_tweets['text'].tolist()
london_text = london_tweets['text'].tolist()

all_text = new_york_text+ paris_text+ london_text
labels = [0] * len(new_york_tweets) + [1] * len(paris_tweets) + [2] * len(london_tweets)

train_x, test_x, train_y, test_y= train_test_split(all_text, labels, train_size = 0.8, random_state = 42)
print(len(train_x), len(test_x))

counter = CountVectorizer()
counter.fit(train_x)
train_counts = counter.transform(train_x)
test_counts = counter.transform(test_x)

print(train_x[3])
print(train_counts[3])

model = MultinomialNB()
model.fit(train_counts, train_y)

predictions = model.predict(test_counts)
accuracy_score = accuracy_score(test_y, predictions)
print(accuracy_score)
cm = confusion_matrix(test_y, predictions)
print(cm)

my_tweet = "Everyone here has got some bad teeth!"
my_counter = counter.transform([my_tweet])
my_prediction = model.predict(my_counter)
print(my_prediction)
