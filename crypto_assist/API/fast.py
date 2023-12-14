from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from crypto_assist.DARTS_predict_copy import load_model, load_vars, smape_function, model_predict, model_predict_accuracy, model_predict, prediction_hist
from crypto_assist.DL_Model_Predict import predict_last_7days, predict_next_7days, DL_load_vars, DL_load_model

app = FastAPI()
app.state.model = load_model()
app.state.vars = load_vars()
app.state.model_DL = DL_load_model()
app.state.vars_DL = DL_load_vars()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define a root '/' endpoint
@app.get('/')
def root():
    return {'greeting':'Hello'}

#Define a predict endpoint for the function
@app.get('/predict')
def predict():
    #Calling model_predict_accuracy and model_predict to get predictions
    smape, actual, past_pred = model_predict_accuracy()
    pred = model_predict()
    df_history, df_val = prediction_hist()
    DL_predict_last = predict_last_7days()
    DL_predict_next = predict_next_7days()


    #Turning the prediction results which are numpy arrays into lists and converting numbers into floats
    actual = [float(item) for item in actual]
    past_pred = [float(item) for item in past_pred]
    pred = [float(item) for item in pred]
    df_history = [float(item) for item in df_history]
    df_val = [float(item) for item in df_val]
    DL_predict_last = DL_predict_last.tolist()
    DL_predict_next = DL_predict_next.tolist()


    return {'smape':float(smape),
            'actual_price_last_5_days':actual,
            'predicted_price_last_5_days': past_pred,
            'predicted_price_for_next_5_days': pred,
            "historical_prediction": df_history,
            "validation":df_val,
            'DL_predict_last': DL_predict_last,
            'DL_predict_next': DL_predict_next
            }
