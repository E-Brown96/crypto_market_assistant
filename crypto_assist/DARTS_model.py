import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import SymmetricMeanAbsolutePercentageError
from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, RobustScaler, StandardScaler
from data_preprocess import preprocessor_not_scaled

from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)
from darts.models import (
    RNNModel,
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
    VARIMA,
)
import logging
logging.disable(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)

#=======================================================================================================================
def DARTS_model():
    def preprocessing():
        ### __*Preprocessing*__
        df = preprocessor_not_scaled('BTCUSDT_daily.csv','final_text_df.csv','social_number_data.csv','FearAndGreedIndex.csv')
        df.index = pd.to_datetime(df.date)
        df.drop(columns="date",inplace=True)
        return df

    def scaling():
        ### __*Scaling and Converting the data*__
        #freq: This parameter defines the frequency or time interval between consecutive data points in the time series.
        #'D': It means that each data point in the time series corresponds to an hour
        timeseries = TimeSeries.from_dataframe(preprocessing(), freq='D')

        # Here we are converting the MinMax scaler to a Time Series version of it
        scaler = MinMaxScaler()
        ts_scaler = Scaler(scaler)
        ts_scaler_target = Scaler(scaler)
        scaled_df = ts_scaler.fit_transform(timeseries)
        ts_scaler_target = ts_scaler_target.fit(timeseries["close"])
        return scaled_df, ts_scaler_target

    def train_test_split():
        X, y = scaling()
        ### __*Train Test Split*__
        train, val = X.split_before(0.8)
        #train.plot(label="training")
        #val.plot(label="validation")
        val,test = val.split_before(0.5)
        return val, test, train

    def earlystop():
        ### __*Setup Earlystopping and the train on GPU*__
        # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
        # a period of 5 epochs (`patience`)
        stopper = EarlyStopping(
            monitor="val_loss",
            patience=30,
            min_delta=0.005,
            mode='min',
        )
        """pl_trainer_kwargs={"callbacks": [stopper],
                        "accelerator": "gpu",
                        "devices": [0]}"""
        return stopper

    def model_instantiate():
        ### __*Instanciate BlockRNNMODEL*__
        # predict 15 days considering the latest 45 days
        model_covariates = BlockRNNModel(
            model="LSTM",
            input_chunk_length=25,
            output_chunk_length=7,
            #dropout=0.2,
            torch_metrics= SymmetricMeanAbsolutePercentageError(),
            n_epochs=100,
            #pl_trainer_kwargs = pl_trainer_kwargs,
            random_state=0,
        )
        return model_covariates

    def train_model():
        ### __*Train the model*__
        val, test, train = train_test_split()
        model_int = model_instantiate().fit(
            series=[train["close"]],    # the target training data
            past_covariates=train,     # the multi covariate features training data
            val_series=[val["close"]],  # the target validation data
            val_past_covariates=val,   # the multi covariate features validation data
            verbose=True,
        )
        return model_int
    scaled_df, ts_scaler_target = scaling()
    val, test, train = train_test_split()
    train_model = train_model()
    return scaled_df, ts_scaler_target, val, test, train, train_model

DARTS_model_vars = DARTS_model()

with open('model.pkl', 'wb') as file:
    pickle.dump(DARTS_model_vars, 'model')

scaled_df, ts_scaler_target, val, test, train, train_model = DARTS_model_vars

def model_predict_accuracy():
    ### __*Do a prediction*__
    pred_cov = train_model.predict(n=7,    # n of days to predict ####
                series=test["close"][-25-7:-7],  # target input for prediction the current week
                past_covariates=test[-25-7:-7])  # past-covariates input for prediction the current week

    ### __*Result of the metrics*__
    # check the SMAPE error

    smape_actual_pred = smape(test['close'][-7:], pred_cov)

    #Real Data
    actual_last_7days = ts_scaler_target.inverse_transform(test['close'][-7:]).values() #Actual last 7 days

    #Predicted Data
    pred_last_7days = ts_scaler_target.inverse_transform(pred_cov).values() #Prediction from last 7 days

    return smape_actual_pred, actual_last_7days, pred_last_7days

def model_predict():
    pred_cov = train_model.predict(n=7,    # n of days to predict ####
                series=test["close"][-25:],  # target input for prediction
                past_covariates=test[-25:])  # past-covariates input for prediction
    pred_7days = ts_scaler_target.inverse_transform(pred_cov).values() #Prediction from last 7 days

    return pred_7days

DARTS_model()
print(model_predict_accuracy())
print(model_predict())
