import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
df = pd.read_csv("/Users/eddie/Desktop/IRIS.csv")

#print(df.head())
print(df.columns)
#print(df.species.values)
#print(df.describe)
#print(df.petal_length.mean())
#print(df.info)

df['species_int'] = df['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2})
#print(df['species_int'])

x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_int']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)

scores = []
for n in range(1,100):
    model = KNeighborsClassifier(n_neighbors = n)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    scores.append(score)
    plt.plot(scores)
plt.title("Accuracy for #Neighbors")
plt.xlabel("# Neighbors")
plt.ylabel("Accuracy")
plt.show()

classifier = KNeighborsClassifier(n_neighbors = 20)
classifier.fit(x_train, y_train)
my_score = classifier.score(x_test, y_test)
print(my_score * 100)

plt.scatter(df['petal_length'], df['species_int'])

my_flower = np.array([1.0, 0.55, 0.01, 2.0])
my_flower = my_flower.reshape(1,-1)
print(my_flower)
my_prediction = classifier.predict(my_flower)

print(my_prediction)
