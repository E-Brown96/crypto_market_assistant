from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import os
from crypto_assist.data_preprocess import preprocessor

#route_path = os.path.dirname(os.path.abspath(__file__))
#print(route_path)

df = preprocessor('BTCUSDT_daily_Binance.csv','final_text_df.csv','social_number_data.csv','FearAndGreedIndex.csv',MinMaxScaler())


print(df)
