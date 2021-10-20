import pandas as pd
import matplotlib.pyplot as plt


stockdata = pd.read_csv("/Users/atsumikitagawa/Desktop/GOOG.csv")

#print(stockdata.head())
closeprice = stockdata['Close']
date = stockdata['Date']
#print(closeprice)
#print(date)

plt.plot(date, closeprice)
plt.xlabel("Years Since 2004")
plt.ylabel("Stock Price")
plt.title("Google Stock Value")
plt.show()
