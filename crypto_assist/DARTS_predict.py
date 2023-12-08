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

with open('DARTS_model.pkl', 'rb') as file:
    DARTS_model_vars = pickle.load(file)

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

print(model_predict_accuracy())
