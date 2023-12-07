import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchmetrics import SymmetricMeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError
from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.datasets import AirPassengersDataset, MonthlyMilkDataset, ElectricityDataset
from darts import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from darts.utils.missing_values import fill_missing_values
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, RobustScaler, StandardScaler
from crypto_assist.data_preprocess import preprocessor_not_scaled

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

### __*Preprocessing*__
df = preprocessor_not_scaled('BTCUSDT_daily_Binance.csv','final_text_df.csv','social_number_data.csv','FearAndGreedIndex.csv')
df.index = pd.to_datetime(df.date)
df.drop(columns="date",inplace=True)


### __*Scaling and Converting the data*__
#freq: This parameter defines the frequency or time interval between consecutive data points in the time series.
#'D': It means that each data point in the time series corresponds to an hour
timeseries = TimeSeries.from_dataframe(df, freq='D')

# Here we are converting the MinMax scaler to a Time Series version of it
from darts.dataprocessing.transformers.scaler import Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
ts_scaler = Scaler(scaler)
ts_scaler_target = Scaler(scaler)
scaled_df = ts_scaler.fit_transform(timeseries)
ts_scaler_target = ts_scaler_target.fit(timeseries["close"])


### __*Cleaning Missing data*__
from darts.utils.missing_values import fill_missing_values
scaled_df = fill_missing_values(scaled_df)


### __*Train Test Split*__
train, val = (scaled_df).split_before(0.8)
#train.plot(label="training")
#val.plot(label="validation")
val,test = (val).split_before(0.5)


### __*Setup Earlystopping and the train on GPU*__
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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


### __*Train the model*__
model_covariates.fit(
    series=[train["close"]],    # the target training data
    past_covariates=train,     # the multi covariate features training data
    val_series=[val["close"]],  # the target validation data
    val_past_covariates=val,   # the multi covariate features validation data
    verbose=True,
)


### __*Do a prediction*__
pred_cov = model_covariates.predict(n=7,                        # n of days to predict
                                 series=test["close"][-25-7:-7],       # target input for prediction
                                 past_covariates=test[-25-7:-7]) # past-covariates input for prediction

test[-7:]["close"].plot(label="actual")
#pred_cov.plot(label="forecast")


### __*Result of the metrics*__
# check the SMAPE error
smape(test['close'][-7:], pred_cov)
#Real Data
ts_scaler_target.inverse_transform(test['close'][-7:]).values()


### __*Turn Prediction into original scale*__
#Predicted Data
ts_scaler_target.inverse_transform(pred_cov).values()


### __*TransformerModel*__
"""
model = TransformerModel(
    input_chunk_length=25,
    output_chunk_length=7,
    n_epochs=20
)
"""

"""
model.fit(target, past_covariates=past_cov)
pred = model.predict(6)
pred.values()
"""

### __*XGBModel!!*__
"""
model = XGBModel(
    lags=25,
    lags_past_covariates=25,
    #lags_future_covariates=[0,1,2,3,4,5],
    output_chunk_length=6,
)
"""

"""
model.fit(val["close"], past_covariates=, future_covariates=)
pred = model.predict(6)
pred.values()

"""
