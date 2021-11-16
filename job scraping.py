import requests
import pandas as pd 
from bs4 import BeautifulSoup

skl = input("Enter a skill you have")
sta = input("Enter the state you live in")
def jobGetter(skill, state):

    url = "https://www.indeed.com/jobs?q={skill}&l={state}&vjk=b80122ef48cfa47d".format(skill = skill, state = state)
    data = requests.get(url)
    soup =  BeautifulSoup(data.text, 'html.parser')

    jobs = soup.find_all('div', class_ = 'slider_container')
    
    for job in jobs:
        company = job.find('span', class_ = 'companyName').text.strip()
        location = job.find('div', class_ = 'companyLocation').text.strip()
        date = job.find('span', class_ = 'date').text.strip()
        descr = job.find('div', class_ = 'job-snippet').li.text.strip()
        #salary = job.find('div', class_= 'attribute_snippet').text.strip()
        job_title = job.find('h2').span.text.strip()
        
        print(f'Company: {company}\n Location: {location}\n Active: {date}\n Decription: {descr}\n Job: {job_title}\n\n')
        
jobGetter(skl, sta)
