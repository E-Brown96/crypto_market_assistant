import os
import shutil
import zipfile
import requests
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
#
#===================SETTINGS=======================#
#Check path for Currency and Dates first!
curreny         = "BTCUSDT"       #BTCUSDT
start_date      = "2017-08-17"    #BTC from  "2017-08-17"
end_date        = "2023-11-30"    #BTC until "2023-11-30"
sample_period   = "D"             #daily
#==================================================#
route_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
csv_filename = route_path+"/raw_data/"+curreny+"_"+sample_period+"_Binance.csv"
foldername = route_path+"/raw_data/"+curreny+"_DL"

#Original Path for Browser:
#https://data.binance.vision/?prefix=data/spot/daily/klines/BTCUSDT/1m/

#DL Path:
base_url = "https://data.binance.vision/data/spot/daily/klines/"+curreny+"/1m/"

# Generate the list of file URLs
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
file_urls = [f"{base_url}{curreny}-1m-{date.strftime('%Y-%m-%d')}.zip" for date in date_range]

# Set the destination folder
destination_folder = Path(foldername) #CHANGE ME
destination_folder.mkdir(parents=True, exist_ok=True)

def check_file_existence(url):
    response = requests.head(url)
    return response.status_code == 200

def download_file(url, destination_path):
    if check_file_existence(url):
        response = requests.get(url)
        with open(destination_path, 'wb') as file:
            file.write(response.content)

# Function to download files in parallel
def download_files_parallel(file_urls, destination_folder):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_file, url, destination_folder / url.split("/")[-1]) for url in file_urls] #DL to Folder/file.zip
        print("Download started...")
        # Wait for all threads to complete
        for future in futures:
            future.result()

# Function to download files in sequential
def download_files_sequential(file_urls, destination_folder):
    for url in file_urls:
        filename = url.split("/")[-1]
        download_file(url, filename)

# Download files in parallel
download_files_parallel(file_urls, destination_folder)
print("---------Download complete----------")

# Unzip files =================================================================#
def unzip_all_files(destination_folder):
    # Ensure the folder path exists
    if not os.path.exists(destination_folder):
        print(f"The folder '{destination_folder}' does not exist.")
        return

    # Get a list of all files in the folder
    files = os.listdir(destination_folder)

    for file in files:
        file_path = os.path.join(destination_folder, file)
        #print(file_path)
        # Check if the file is a zip file
        if file.endswith('.zip'):
            try:
                # Create a ZipFile object
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract all contents to the folder
                    zip_ref.extractall(destination_folder)

                # print(f"Unzipped: {file}")
            except zipfile.BadZipFile:
                print(f"Skipped: {file} (Not a valid zip file)")
    print("---------Unzip complete----------")

#Call
unzip_all_files(destination_folder)

# Delete files ================================================================#
# Count the number of .csv and .zip files in the destination folder
csv_files = list(destination_folder.glob("*.csv"))
zip_files = list(destination_folder.glob("*.zip"))
print(f"Number of .csv files: {len(csv_files)}")
print(f"Number of .zip files: {len(zip_files)}")

if len(csv_files) == len(zip_files):
  for zip_file in zip_files:
      zip_file.unlink()
  print("ZIP files deleted")
else:
  print("Error: Amount of CSV files not same than ZIP files.")

# Clean dataframes ============================================================#
CSV_df = []
for csv_file in csv_files:
  _data = pd.read_csv(csv_file, names=["time", "open", "high", "low", "close", "volume BTC", "close_time","volume USD","num_trades","taker_buy_volume","taker_buy_quote_volume","ignore"])

  #Format and clean data
  _data["time"] = pd.to_datetime(_data["time"], unit="ms")
  _data = _data.sort_values(by='time')
  _data.set_index("time", inplace=True)
  _data = _data.drop(columns=["close_time", "num_trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"])
  CSV_df.append(_data)

# Merge and Resample ==========================================================#
merged_df = pd.concat(CSV_df).sort_values(by="time")
merged_df = merged_df.resample(sample_period).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume BTC': 'sum',
    'volume USD': 'sum'
})
# Export ======================================================================#
merged_df.to_csv(csv_filename) #Filename defined in the settings
print("File created: "+csv_filename)

# Remove temp files ===========================================================#
#Check if new CSV file exist and zipped files was ok.
if len(csv_files) == len(zip_files) and Path(csv_filename).exists():
  shutil.rmtree(destination_folder)
  print("Temp files removed")
else:
  print("Error: Temp files not removed!")
