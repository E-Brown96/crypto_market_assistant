from darts.metrics import mape, smape, mae
import pickle
from darts.models import BlockRNNModel
import torch
import pickle
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import SymmetricMeanAbsolutePercentageError
import numpy as np

def load_model():

    # Load the BlockRNNModel with MPS support
    train_model = BlockRNNModel.load("DARTS_model.pkl", map_location="cpu")
    return train_model

def load_vars():
    #Load from pickle file
    with open('DARTS_vars.pkl', 'rb') as file:
        DARTS_model_vars = pickle.load(file)
    scaled_df, ts_scaler_target, val, test, train = DARTS_model_vars
    return scaled_df, ts_scaler_target, val, test, train


def smape_function(A, F):
    # check the SMAPE error
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def model_predict_accuracy():
    train_model = load_model()
    scaled_df, ts_scaler_target, val, test, train = load_vars()
    ### __*Do a prediction*__
    pred_cov = train_model.predict(n=5,    # n of days to predict ####
                series=test["close"][-35-5:-5],  # target input for prediction the current week
                past_covariates=test[-35-5:-5],
                )  # past-covariates input for prediction the current week

    ### __*Result of the metrics*__

    #Real Data
    actual_last_7days = ts_scaler_target.inverse_transform(test['close'][-7:]).values() #Actual last 7 days

    #Predicted Data
    pred_last_7days = ts_scaler_target.inverse_transform(pred_cov).values() #Prediction from last 7 days

    # check the SMAPE error
    #smape_actual_pred = smape(test['close'][-7:], pred_cov)
    smape_actual_pred = smape_function(actual_last_7days, pred_last_7days)

    return smape_actual_pred, actual_last_7days, pred_last_7days


def model_predict():
    train_model = load_model()
    scaled_df, ts_scaler_target, val, test, train = load_vars()
    pred_cov = train_model.predict(n=5,    # n of days to predict ####
                series=test["close"][-35:],  # target input for prediction
                past_covariates=test[-35:],
                )  # past-covariates input for prediction
    pred_7days = ts_scaler_target.inverse_transform(pred_cov).values() #Prediction from last 7 days

    return pred_7days
