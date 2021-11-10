from bs4 import BeautifulSoup
import requests
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib as plt
from csv import writer

url = "https://www.redfin.com/city/18484/NJ/Tenafly/filter/viewport=40.92572:40.9064:-73.88865:-74.01568"

data = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
soup= BeautifulSoup(data.text, 'html.parser')
print (data) 

homes = soup.find_all('div', class_ = "bottomV2")

with open('housing1.csv', 'w', encoding = 'utf8', newline='') as f:
    my_writer = writer(f)
    header = ["Price", "Beds", "Address", "Info"]
    my_writer.writerow(header)
    for home in homes:
        price = home.find('span', class_ = 'homecardV2Price').text.strip()
        beds = home.find('div', class_ = 'stats').text.strip()
        address = home.find('span', attrs = {'data-rf-test-id':'abp-streetLine'}).text.strip()
        info = home.find('div', class_ = "disclaimerV2").text.strip()
        #link = home.find('a', attrs = {'href': re.compile('')})
        info = [price, beds, address, info]
        my_writer.writerow(info)
        print(price, beds, address, info)
