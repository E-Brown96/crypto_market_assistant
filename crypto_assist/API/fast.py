from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from crypto_assist.DARTS_predict_copy import load_model, load_vars, smape_function, model_predict, model_predict_accuracy, model_predict


app = FastAPI()
app.state.model = load_model()
app.state.vars = load_vars()

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


    #Turning the prediction results which are numpy arrays into lists and converting numbers into floats
    actual = [float(item) for item in actual]
    past_pred = [float(item) for item in past_pred]
    pred = [float(item) for item in pred]

    return {'smape':float(smape),
            'actual_price_last_5_days':actual,
            'predicted_price_last_5_days': past_pred,
            'predicted_price_for_next_5_days': pred,
            }
