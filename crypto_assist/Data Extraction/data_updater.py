from request_social_2 import check_social_data
from request_fearandgreed_2 import fear_and_greed
from request_sentiment_2 import sentimental_data
from request_binance_new import chart_data

def update_data():
    #Check and update chart data
    chart_data()

    #Check and update social data
    check_social_data()

    #Check and update fear and greed data
    fear_and_greed()

    #Check and update sentimental data
    sentimental_data()

    print('✅ All dataframes have been updated.')
