import pandas as pd
import pandas_ta as pta
import numpy as np
import os
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder

#The first part of the preprocessing stage is to calculate TA indicators on the chart data

def ta_indicators(file_BTC:str):
    #Create route path and extract the BTC dataframe
    route_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BTC = pd.read_csv(os.path.join(route_path, 'raw_data',file_BTC))

    #Create the RSI class
    RSI = pta.rsi(BTC['close'], length = 14)    #Calc RSI
    RSI = RSI.replace(np.NaN, 50)               #clean NAN values
    BTC["RSI"] = RSI

    # Add RSI Class to df
    # [-1] Oversold <= 30
    # [+1] Overbought >= 70
    # [0]  Neutral 30-70
    BTC['RSI_class'] = np.where(RSI > 70, 1, np.where(RSI < 30, -1, 0))

    #Calculate ADX and create ADX class
    ADX_df = pta.adx(BTC['high'], BTC['low'], BTC['close'], length=14) #Calc ADX
    ADX = ADX_df.drop(columns=["DMP_14", "DMN_14"]) #Clean up columns
    ADX = ADX.replace(np.NaN, 0)                    #Clean NaN values
    BTC["ADX"] = ADX                                #Add RSI column to BTC df

    # Add ADX_Class to df
    #[1] 0 – 25	    Absent or Weak Trend
    #[2] 25 – 50	Strong Trend
    #[3] 50 – 75	Very Strong Trend
    #[4] 75 – 100	Extremely Strong Trend
    BTC['ADX_class'] = np.where(ADX <= 25, 1,
                        np.where(ADX <= 50, 2,
                            np.where(ADX <= 75, 3,
                                4)))
    print(BTC)
    return BTC

ta_indicators('BTCUSDT_daily.csv')


'''
Inputs need to be the the filename of the saved csv files, must be strings ending with .csv:
1 Crypto Chart Data
2 Sentimental Data
3 Social Data
4 Fear and Greed Index Data'''


def data_merging(file_BTC:str,file_Sentimental:str,file_Social:str,file_FAGI:str):
    #Create the route path so it can be used on any machine
    route_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    #Extract the dataframes from Raw Data folder using filenames
    df_sentimental = pd.read_csv(os.path.join(route_path, 'raw_data',file_Sentimental))
    df_social = pd.read_csv(os.path.join(route_path, 'raw_data',file_Social))
    df_binance = file_BTC
    df_fear_and_greed = pd.read_csv(os.path.join(route_path, 'raw_data',file_FAGI))

    #The chart data makes a dataframe where the first column is called time,
    #we need to replace that with date
    df_binance.rename(columns={'time': 'date'}, inplace=True)

    #Now merge the dataframes, left merging onto the chart data.
    df_merge = pd.merge(df_binance, df_fear_and_greed,how='left', on='date')
    df_merge = pd.merge(df_merge, df_sentimental, how='left', on='date')
    df_merge = pd.merge(df_merge, df_social, how='left', on='date')

    print("✅ Data has been merged")

    #Return the merged dataframe
    return df_merge

def imputing(df_merge: pd.DataFrame):
    '''Now we are going to impute the empty cells. There is a lack
    of data pre 2020 in sentimental, social and FAGI so multiple
    methods to impute zero values are used.'''

    #Imputing sentimental data
    sentiment_columns = ['scored_title','scored_text','average_score']

    '''The first process will be the sentimental data columns scored_title,
    scored_text and average_score. These are already scaled but empty values
    need to be processed. Given the empty values are due to no news articles
    this is the equivalent to news sentiment being completely neutral,
    or equal to score in the case of a score. So all empty values are input
    with a score of 0.'''
    df_merge[sentiment_columns] = df_merge[sentiment_columns].fillna(0)

    #Imputing social data
    social_columns = ['twitter_followers','reddit_active_users','reddit_comments_per_day', 'twitter_favourites','reddit_subscribers']

    '''Social data columns are; twitter_followers, twitter_favourites,
    reddit_subscribers, reddit_active_users, reddit_comments_per_day.
    The active users on reddit, twitter favourites and the comments per
    day will have the mean applied. The other two coluns follow a linear
    pattern but have some missing elements so to process this the data
    will be inputed linearly and then finalised using a KNN imputer.'''

    #First replace 0 values with NaN
    df_merge[social_columns] = df_merge[social_columns].replace(0, np.nan)

    #Impute the mean social columns
    columns_to_impute_mean = ['reddit_active_users','reddit_comments_per_day', 'twitter_favourites']

    #Mean imputer
    imputer_mean = SimpleImputer(strategy='mean')

    #Impute mean
    df_merge[columns_to_impute_mean] = imputer_mean.fit_transform(df_merge[columns_to_impute_mean])

    #Now start the twitter follows and reddit subsrcibers linear imputing by making the first values 1000
    df_merge.at[0, 'twitter_followers'] = 1000
    df_merge.at[0, 'reddit_subscribers'] = 1000

    #KNN imputer
    imputer_knn = KNNImputer(n_neighbors=5)

    columns_to_kNN_impute = ['twitter_followers','reddit_subscribers']

    #Linear interpolate the values to begin
    df_merge[columns_to_kNN_impute] = df_merge[columns_to_kNN_impute].interpolate(method='linear')

    #KNN fit and transform
    df_merge[columns_to_kNN_impute] = imputer_knn.fit_transform(df_merge[columns_to_kNN_impute])

    #Imputing fear and greed index
    '''For this data any NaN values will be replace with the Neutral value'''

    df_merge['FAGI_sentiment'] = df_merge['FAGI_sentiment'].replace(np.nan, 'Neutral')

    # Instantiate the Ordinal Encoder
    ordinal_encoder = OrdinalEncoder(categories = [["Extreme Fear","Fear","Neutral","Greed", "Extreme Greed"]])

    # Fit it
    ordinal_encoder.fit(df_merge[["FAGI_sentiment"]])

    # Transforming categories into ordered numbers
    df_merge["FAGI_sentiment_encoded"] = ordinal_encoder.transform(df_merge[["FAGI_sentiment"]])

    #Replace the NaN FAGI scores with the neutral score of 50
    df_merge['FAGI_score'] = df_merge['FAGI_score'].replace(np.nan, 50)

    #Drop the FAGI_sentiment column
    df_merge.drop(columns='FAGI_sentiment', inplace=True)

    #Replace any leftover numerical column values that are NaN with a 0
    df_merge = df_merge.replace(np.NaN, 0)

    print("✅ Data has been imputed")

    return df_merge

def data_scaling(df_merge: pd.DataFrame, scaler):
    '''The final preprocessing step is to scale the data.
    Note a scaler must be input'''

    #Select all columns that are not date
    numeric_columns = df_merge.select_dtypes(include=['float', 'int']).columns

    #Create an instance of the scaler
    the_scaler = scaler

    #Scale numeric columns excluding the date column
    df_merge[numeric_columns] = scaler.fit_transform(df_merge[numeric_columns])

    print("✅ Data has been scaled")

    return df_merge

def preprocessor(file_BTC:str,file_Sentimental:str,file_Social:str,file_FAGI:str, scaler):

    #Call TA indicators
    df_BTC = ta_indicators(file_BTC=file_BTC)

    #Call data merging
    df_merge = data_merging(file_BTC=df_BTC,file_Sentimental=file_Sentimental,file_Social=file_Social,file_FAGI=file_FAGI)

    #Call imputer
    df_merge = imputing(df_merge=df_merge)

    #Call the scaler
    df_merge = data_scaling(df_merge=df_merge, scaler=scaler)

    print("✅ Data has been processed")

    #Return processed data
    return df_merge

def preprocessor_not_scaled(file_BTC:str,file_Sentimental:str,file_Social:str,file_FAGI:str):

    #Call TA indicators
    df_BTC = ta_indicators(file_BTC=file_BTC)

    #Call data merging
    df_merge = data_merging(file_BTC=df_BTC,file_Sentimental=file_Sentimental,file_Social=file_Social,file_FAGI=file_FAGI)

    #Call imputer
    df_merge = imputing(df_merge=df_merge)

    print("✅ Data has been processed")

    #Return processed data
    return df_merge
