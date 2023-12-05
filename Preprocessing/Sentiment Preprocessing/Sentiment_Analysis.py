#Python file for the sentiment analysis

#imports
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import os

def sentiment_processor():
    #Step 1: Load the datasets
    route_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    df_2019 = pd.read_csv(os.path.join(route_path,'Raw_Data','2019_2021_data.csv'))
    df_2021 = pd.read_csv(os.path.join(route_path,'Raw_Data','2021_2022_data.csv'))
    df_2022 = pd.read_csv(os.path.join(route_path,'Raw_Data','sentimental_data.csv'))
    #df_2019 = pd.read_csv('../../Raw_Data/2019_2021_data.csv')
    #df_2021 = pd.read_csv('../../Raw_Data/2021_2022_data.csv')
    #df_2022 = pd.read_csv('../../Raw_Data/sentimental_data.csv')

    #Step 2: Remove unneeded columns

    #First for df_2019
    df_2019 = df_2019.drop(columns=['cryptocurrency','url','predicted_labels','sentiment','subjectivity'])
    df_2019 = df_2019.rename(columns = {'polarity':'sentiment'})

    #Second for df_2021
    df_2021 = df_2021.drop(columns=['source','subject','url'])

    #Then for df_2022
    df_2022.rename(columns = {'summary':'text','sentiment_score':'sentiment'}, inplace=True)

    #Step 3: Converting all the datetime values to the same format yyyy-mm-dd

    #For df_2019
    df_2019['date'] = pd.to_datetime(df_2019['date'].str[:-6])
    df_2019['date'] = df_2019['date'].dt.strftime('%Y-%m-%d')

    #For df_2021
    df_2021['date'] = pd.to_datetime(df_2021['date']).dt.strftime('%Y-%m-%d')

    #For df_2022
    df_2022['date'] = df_2022['date'].apply(lambda x: datetime.strptime(x, "%Y%m%dT%H%M%S").strftime('%Y-%m-%d'))

    #Step 4: Rearranging all the columns so the tables can be concatenated
    column_order = ['date','title','text','sentiment']

    df_2019 = df_2019[column_order]
    df_2021 = df_2021[column_order]
    df_2022 = df_2022[column_order]

    #Now concatenate the dataframes
    combined_df = pd.concat([df_2019,df_2021,df_2022])

    #Then sort the values by date
    sentimental_df = combined_df.sort_values('date').reset_index()

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

print(sentiment_processor())
