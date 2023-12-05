from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from Preprocessing.data_preprocess import preprocessor

df = preprocessor('BTCUSDT_daily_Binance.csv','final_text_df.csv','social_number_data.csv','FearAndGreedIndex.csv',MinMaxScaler())

print(df)
