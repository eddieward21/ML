from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
digits = datasets.load_digits()
#print(digits.data)
#print(digits.images[0])
#print(digits.target)

print(len(digits.data))
pl.gray()
pl.matshow(digits.images[19])
pl.show()

print(digits.images[19])


x = digits.images.reshape(1797, -1)
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x,y)
print(x_train.shape)

model = SVC(kernel='linear', degree=3, gamma = 'scale')
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print(score)

print(classification_report(y_predict, y_test))
print(confusion_matrix(y_predict, y_test))
