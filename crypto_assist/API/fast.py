from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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

@app.get('/predict')
def predict():
    return {'wait':64}


'''@app.get('/predict')
def predict(day_of_week, time):
    # Compute `wait_prediction` from `day_of_week` and `time`
    wait_prediction = int(day_of_week) * int(time)

    return {'wait': wait_prediction}'''
