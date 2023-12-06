import requests
import pandas as pd
import os

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


limit = 0 #0 = every day // 10 = just last 10
FAGI_data = get_fear_and_greed_index(limit)

FAGI_data = pd.DataFrame(FAGI_data)
FAGI_data = FAGI_data.drop(columns=["time_until_update"]) #Drop
FAGI_data = FAGI_data.rename(columns={'timestamp' : 'date', 'value': 'FAGI_score', 'value_classification': 'FAGI_sentiment'}) #rename
FAGI_data = FAGI_data[['date', 'FAGI_score', 'FAGI_sentiment']] #Sort
FAGI_data['date'] = pd.to_datetime(FAGI_data["date"], unit="s")
FAGI_data.set_index("date", inplace=True)
FAGI_data = FAGI_data.sort_values(by="date")

route_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FAGI_data.to_csv(os.path.join(route_path, 'Raw_Data','FearAndGreedIndex.csv')) #Filename defined in the settings
print("File created: FearAndGreedIndex.csv")
