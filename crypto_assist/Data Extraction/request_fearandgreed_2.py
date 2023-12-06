import requests
import pandas as pd
from datetime import datetime, timedelta
import os

def fear_and_greed():
    #This defines the current day
    today = datetime.now().strftime("%Y-%m-%d")

    date_format = '%Y-%m-%d'

    # Convert the string to a datetime object
    today_date = datetime.strptime(today, date_format)

    #Now collect the last date in the csv_file social_number_data
    route_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    df_FAGI = pd.read_csv(os.path.join(route_path, 'raw_data','FearAndGreedIndex.csv'))

    last_date = df_FAGI['date'].iloc[-1]

    #Convert this last date into a date
    last_date_object = datetime.strptime(last_date, date_format)

    #Finding days to fetch
    def days_to_fetch(start=today_date, final=last_date_object):
        if start == final:
            return start
        else:
            return (start - final).days

    #This is the request to the API
    def get_fear_and_greed_index(limit=1):
        url = f"https://api.alternative.me/fng/?limit={limit}"

        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
            FAGI = response.json()

            if 'data' in FAGI: #Data check
                return FAGI['data']
            else:
                print("Error: No data found.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None

    #This will check if the last date in the dataframe equals todays date and if it doesn't will fetch and append the latest date.
    def date_check(days_to_fetch=days_to_fetch(), start=today_date):
        if days_to_fetch == start:
            print("✅ Fear and greed dataframe is up to date.")
            return
        else:
            limit = days_to_fetch #0 = every day // 10 = just last 10
            FAGI_data = get_fear_and_greed_index(limit)

            FAGI_data = pd.DataFrame(FAGI_data)
            FAGI_data = FAGI_data.drop(columns=["time_until_update"]) #Drop
            FAGI_data = FAGI_data.rename(columns={'timestamp' : 'date', 'value': 'FAGI_score', 'value_classification': 'FAGI_sentiment'}) #rename
            FAGI_data = FAGI_data[['date', 'FAGI_score', 'FAGI_sentiment']] #Sort
            FAGI_data['date'] = pd.to_datetime(FAGI_data["date"], unit="s")
            FAGI_data.set_index("date", inplace=True)
            FAGI_data = FAGI_data.sort_values(by="date")

            FAGI_data.to_csv((os.path.join(route_path, 'raw_data','FearAndGreedIndex.csv')), mode='a', header=False, index=True)

            print("✅ File created: FearAndGreedIndex.csv")
    date_check()
