<h2>Welcome to the Crypto Market Assistant Project!</h2>

This project was completed as the final project for a 9 week intensive bootcamp with Le Wagon as part of a team of 4. 

<h3>Aims</h3>
The main aim of the project was to predict the price of the cryptocurrency Bitcoin for the next 5 days using deep learning models. 

<h3>Overview</h3>
The project uses 4 dataframes that were merged together these include:
- The real time chart data of Bitcoin from 2017 till the present day (or last day the data was updated!)
- The sentimental data from news articles related to crypto
  - To develop a sentiment score for this I sourced a BERT language processing model from hugging face trained on a dataset of crypto tweets and used this to generate a sentiment score for each article
- The fear and greed data, the number for the fear or greed for that day dating back to 2019
- The social media data including twitter followers, reddit subscribers, twitter favourites, reddit comments per day, reddit active users per day

<h3>Updating the data sources</h3>
I created a python package within the crypto assist folder called data_updater.py and when ran it will fetch from your data tables the last date you have. 
It then checks this against the current date and if any days are missing it will make a request to each API for each data set and fetch the required data appending it to the csv files.
The file also tells you if your tables are up to date by printing out a message to the console.

<h3>Cleaning the data</h3>
I created the python package that cleans the data sources which again can be found in the crypto_assist folder and is in a file called data_preprocess.py.
This does a number of things it imputes the missing values for the data, it merges the dataframes together, it creates technical indicators based off the chart data and it scales the data. However there is a function you can call where the data is unscaled and one if the data is scaled.

<h3>Modelling</h3>
The model testing was done in the jupyter files called DARTS testing and LSTM testing. We choose to go with the DARTS library for one type of model and also an LSTM model for the other type with one team member working on each.
Both team members used jupyter notebooks to get the csv files and apply the models so they could alter parameters easily. Once this was done I python packaged both the DARTS model (in files called DARTS_model.py & DARTS_predict.py) and the LSTM model (In a file called & DL_Model_Train.py DL_Model_Predict.py)
Note to use the DARTS library on a MAC with an M1 or M2 processor you have to run the file DARTS_model_mac.py as the parameters are changed to use a macs CPU other wise it causes an error.
Each python training file trains the model on the dataset and saves this trained model as a pickel file. The predict python files then call this pickle file and use it to predict the price pf Bitcoin.

<h3>Creating an API</h3>
I also created an API to host the prediction from both models, and this was done using Docker (the docker code can be found in DockerFile) and a fast.py file to call the required python functions.
Note if you wish to run the API yourself note that you need to set up google cloud credentials and a google cloud account. These are configured to for google cloud credentials saved onto my Zsh for security reasons.
From the API you can call both predictions for the next five days as well as predictions for the current five days to check accuracy and historical predictions for the last six months.

<h3>Front End</h3>
The front end was created using Streamlit and can be found in the crypto_assist_UI repo on my Github.

<h3>How to run</h3>
Should you wish to use the files please get in touch, as the data files need to be downloaded seperately.
