FROM python:3.10.6-buster

# First, pip install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Then only, install crypto_assist!
COPY crypto_assist crypto_assist
COPY setup.py setup.py
RUN pip install .

CMD uvicorn crypto_assist.API.fast:app --host 0.0.0.0 --port $PORT
