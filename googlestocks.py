import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

data = pd.read_csv("/Users/atsumikitagawa/Desktop/honeyproduction.csv")
#print(data.head())

prod_in_year = data.groupby('year').totalprod.mean().reset_index()
#print(prod_in_year)

x = prod_in_year['year']
x= x.values.reshape(-1, 1)
y = prod_in_year['totalprod']

#print(x)
#print(y)

plt.scatter(x,y)

regr = LinearRegression()
regr.fit(x,y)
coefficient = regr.coef_
intercept = regr.intercept_
#print(coefficient[0])
#print(intercept)

y_prediction = regr.predict(x)
#print(y_prediction)

plt.plot(x,y_prediction)
plt.xlabel("Year")
plt.ylabel("Pounds of Honey in Millions")
plt.title("Honey Production 1998-2035")

x_future = np.array(range(2012, 2036))
x_future = x_future.reshape(-1,1)
#print(x_future_prediction)

y_future_prediction = regr.predict(x_future)
plt.plot(x_future, y_future_prediction)
plt.show()
