from darts.metrics import mape, smape, mae
import pickle
from darts.models import BlockRNNModel

with open('DARTS_vars.pkl', 'rb') as file:
    DARTS_model_vars = pickle.load(file)
scaled_df, ts_scaler_target, val, test, train = DARTS_model_vars

train_model = BlockRNNModel.load("DARTS_model.pkl")

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
    print(type(train_model))
    pred_cov = train_model.predict(n=7,    # n of days to predict ####
                series=test["close"][-25:],  # target input for prediction
                past_covariates=test[-25:])  # past-covariates input for prediction
    pred_7days = ts_scaler_target.inverse_transform(pred_cov).values() #Prediction from last 7 days

    return pred_7days

print(model_predict())
