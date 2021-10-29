import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

income_data = pd.read_csv("adult.csv")
print(income_data.iloc[0])

income_data["sex-int"] = income_data["sex"].apply(lambda row:1 if row == "Male" else 0)
income_data["country-int"] = income_data['native.country'].apply(lambda row: 1 if row == "United States" else 0)

X = income_data[["age", "capital.gain", "capital.loss", "hours.per.week", "sex-int", "country-int"]]
y = income_data[["income"]]

train_X, test_X, train_y, test_y  = train_test_split(X, y, random_state=1)

forest = RandomForestClassifier(random_state= 1)
forest.fit(train_X, train_y)
score = forest.score(test_X, test_y)
print(score)
