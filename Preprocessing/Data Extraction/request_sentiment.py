import requests
import numpy as np
import pandas as pd
import time
import datetime

#First step is to get the data from the API, I am using a for loop to make a request every month for news related to Bitcoin.
all_data = []
initial_date = 20220301
current_date = 20220301
current_date_2 = 20220401
for i in range(0,24):
    date = str(current_date)
    date_2 = str(current_date_2)
    url= f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=COIN,CRYPTO:BTC&time_from={date}T0000&time_to={date_2}T0000&limit=1000&apikey=LYY7J4L4FM16HORC'
    r = requests.get(url)
    all_data.append(r.json())
    today = datetime.datetime.now().strftime("%Y%m%d")
    time.sleep(2)
    if int(current_date) < 20221201 and int(current_date_2) < 20221201:
        current_date += 100
        current_date_2 += 100
    elif int(current_date_2) == 20221201:
        current_date_2 = 20230101
        current_date += 100
    elif int(current_date) == 20221201:
        current_date_2 += 100
        current_date = 20230101
    elif int(current_date_2) < 20231201:
        current_date += 100
        current_date_2 += 100
    else:
        current_date = 20231201
        current_date_2 = today

#After collecting the data from the API I am now extracting the information from the json file, for each created monthly list
feed = []
for i in range(0,len(all_data)):
    #Within the json file is a list and a dictionary with key 'feed'
    #So for each month [i] I am extracting the dict values for the key 'feed'
    feed.append(all_data[i]['feed'])

#Now I am extracting the exact information I want for each value from the feed
updated_data = []
for j in feed:
    for k in j:
        #I am getting the time, title, summary and the sentiment score for each news article
        updated_data.append((k['time_published'],k['title'],k['summary'],k['ticker_sentiment'][0]['ticker_sentiment_score']))
#Then I convert the updated_data list to a numpy array
numpy_list = np.array(updated_data)

#The numpy array is converted to a pandas dataframe and the columns given appropriate titles
df = pd.DataFrame(numpy_list)
df.columns = ['date', 'title', 'summary', 'sentiment_score']

#The final dataframe is organised by date, the index is reset and alternative index removed
final_df = df.sort_values(by='date', ascending=True).reset_index().drop(columns='index')

#Finally the dataframe is uploaded to a csv file
final_df.to_csv('sentimental_data.csv', index=False)
