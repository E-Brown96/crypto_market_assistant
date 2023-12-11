from darts.metrics import mape, smape, mae
import pickle
from darts.models import BlockRNNModel
import torch
import pickle
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import SymmetricMeanAbsolutePercentageError

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

def model_predict_accuracy():
    train_model = load_model()
    scaled_df, ts_scaler_target, val, test, train = load_vars()
    ### __*Do a prediction*__
    pred_cov = train_model.predict(n=7,    # n of days to predict ####
                series=test["close"][-25-7:-7],  # target input for prediction the current week
                past_covariates=test[-25-7:-7],
                )  # past-covariates input for prediction the current week

    ### __*Result of the metrics*__
    # check the SMAPE error

    smape_actual_pred = smape(test['close'][-7:], pred_cov)

    #Real Data
    actual_last_7days = ts_scaler_target.inverse_transform(test['close'][-7:]).values() #Actual last 7 days

    #Predicted Data
    pred_last_7days = ts_scaler_target.inverse_transform(pred_cov).values() #Prediction from last 7 days

    return smape_actual_pred, actual_last_7days, pred_last_7days


def model_predict():
    train_model = load_model()
    scaled_df, ts_scaler_target, val, test, train = load_vars()
    pred_cov = train_model.predict(n=7,    # n of days to predict ####
                series=test["close"][-25:],  # target input for prediction
                past_covariates=test[-25:],
                )  # past-covariates input for prediction
    pred_7days = ts_scaler_target.inverse_transform(pred_cov).values() #Prediction from last 7 days

    return pred_7days

print(model_predict_accuracy())
print(model_predict())
