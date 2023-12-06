import requests
import numpy as np
import pandas as pd
import time
import os
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from datetime import datetime, timedelta

def sentimental_data():
    #This defines the current day
    today = datetime.now().strftime("%Y-%m-%d")

    date_format = '%Y-%m-%d'

    # Convert the string to a datetime object
    today_date = datetime.strptime(today, date_format)

    #Now collect the last date in the csv_file social_number_data
    route_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    df_FAGI = pd.read_csv(os.path.join(route_path, 'raw_data','final_text_df.csv'))

    last_date = df_FAGI['date'].iloc[-1]

    #Convert this last date into a date
    last_date_object = datetime.strptime(last_date, date_format)

    last_date_for_fetch = last_date_object + timedelta(days=1)

    formatted_last_date = last_date_for_fetch.strftime("%Y%m%d")

    #Finding days to fetch
    def days_to_fetch(start=today_date, final=last_date_object):
        if start == final:
            return start
        else:
            return (start - final).days

    def date_check(days_to_fetch=days_to_fetch(), start=today_date):
        if days_to_fetch == start:
            print("✅ Sentimental dataframe is up to date.")
            return
        else:
            url= f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=COIN,CRYPTO:BTC&time_from={formatted_last_date}T0000&limit=1000&apikey=LYY7J4L4FM16HORC'
            r = requests.get(url)
            data = r.json()

            feed = []
            for i in range(0,len(data)):
                feed.append(data['feed'])

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

            print("✅ Sentimental values collected from API.")

            def sentiment_processor(df_2022=final_df):
                #Step 2: Remove unneeded columns

                #Then for df_2022
                df_2022.rename(columns = {'summary':'text','sentiment_score':'sentiment'}, inplace=True)

                #Step 3: Converting all the datetime values to the same format yyyy-mm-dd

                #For df_2022
                df_2022['date'] = df_2022['date'].apply(lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S").strftime('%Y-%m-%d'))

                #Step 4: Rearranging all the columns so the tables can be concatenated
                column_order = ['date','title','text','sentiment']

                df_2022 = df_2022[column_order]

                #Then sort the values by date
                sentimental_df = df_2022.sort_values('date').reset_index()

                #Ginally perform a sentimental analysis using the kk08/CryptoBERT model which was trained on crypto data
                #Importing the tokenizer and model
                tokenizer = BertTokenizer.from_pretrained("kk08/CryptoBERT")
                model = BertForSequenceClassification.from_pretrained("kk08/CryptoBERT")

                #Creating a pipeline for the sentiment analysis called classifier

                #Note this model returns a sentiment score and either LABEL 0 if the score is negative or LABEL 1 if the score is postive
                classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

                #Then apply the pipeline to the title and text columns of our dataframe
                sentimental_df['title_prediction'] = sentimental_df['title'].apply(lambda x: classifier(x))
                sentimental_df['text_prediction'] = sentimental_df['text'].apply(lambda x: classifier(x))

                #Next create a function that will turn all results for LABEL 0 to negative numbers so they can be input into our final model correctly
                def map_to_score(row):
                    label = row[0]['label']
                    score = row[0]['score']

                    if label == 'LABEL_1':
                        return score
                    else:
                        return -score

                #Now create a copy of the dataframe
                scored_df = sentimental_df.copy()

                #Then apply the map_to_score function
                scored_df['scored_title'] = scored_df['title_prediction'].apply(map_to_score)
                scored_df['scored_text'] = scored_df['text_prediction'].apply(map_to_score)

                #We then create a final column which is the average sentiment score of the title and text columns
                scored_df['average_score'] = (scored_df['scored_title'] + scored_df['scored_text'])/2

                #Select the columns we want to use
                processed_text_df = scored_df[['date','scored_title','scored_text','average_score']]

                #Create a final df that finds the mean sentiment score for each day
                final_text_df = processed_text_df.groupby('date', as_index=False).mean()

                #Return the final dataframe with the following columns: date, scored_title, scored_text and average_score
                return final_text_df
            print("✅ Sentimental data analysed.")
            final_df = sentiment_processor()
            final_df.to_csv((os.path.join(route_path, 'raw_data','final_text_df.csv')), mode='a', header=False, index=False)
            print("✅ Sentimental data has been added to the dataframe, dataframe up to date.")
    date_check()
