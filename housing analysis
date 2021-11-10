import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('/Users/eddie/Desktop/housing1.csv')
#print(df.head())
#print(df.columns)
#print(df['Address'].dtype)
#df.describe()
x = df["Beds"]
x[3] = '4'
x=x.str[0]
x = pd.to_numeric(x)
#print(x.dtype)
#print(x)
#thank god. Mwah

y = df['Price']
y = y.str.replace(',', '',).str[1:]
y = pd.to_numeric(y)
#print(y.dtype)
#YESS!
#print(y)
mean = round(y.mean())
print(f'Mean Price of House in Tenafly: {mean}')
avg_beds = (round(x.mean()))
print(f'Average Number of Beds: {avg_beds}')
model = LinearRegression()
#print(x.ndim)
#print(y.ndim)
#print(x.shape)
#print(y.shape)

x= x.values.reshape(-1,1)
#print(x.shape, x.ndim)
y = y.values.reshape(-1,1)
#print(y.shape, y.ndim)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8)
model.fit(x_train, y_train)
model_prediction = model.predict(x_test)
#print(model_prediction)
#print(y_test)
print(model.score(x_train, y_train))
#print(model.coef_)
#print(model.intercept_)

my_house = np.array([3])
joe_house = np.array([11])
stu_house = np.array([2])

our_houses = np.array([my_house, joe_house, stu_house])
our_prediction = model.predict(our_houses)
print(our_prediction)

plt.scatter(x,y)
plt.xlabel("Number of Beds")
plt.ylabel("House Price in $")
plt.title("House Price Based on # of Beds in Tenafly")
plt.show()
plt.plot(x_test, model_prediction)
plt.show()
plt.plot(y_test, model_prediction)

plt.show()
