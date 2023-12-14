FROM python:3.10.6-buster

WORKDIR /test

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Then only, install crypto_assist!
COPY crypto_assist crypto_assist
COPY setup.py setup.py
COPY DARTS_model.pkl .
COPY DARTS_vars.pkl .
COPY DARTS_model.pkl.ckpt .
COPY DL_Model_Trained.keras .
COPY DL_vars.pkl .

RUN pip install .

CMD uvicorn crypto_assist.API.fast:app --host 0.0.0.0 --port $PORT
