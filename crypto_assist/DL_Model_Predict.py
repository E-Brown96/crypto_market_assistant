import tensorflow as tf
from keras import models
import pickle
#from crypto_assist.DL_Model_Train import smape


def smape(y_true, y_pred):
        return tf.reduce_mean(((abs(y_true - y_pred)) / ((y_true + y_pred)/2))* 100, axis=-1)

def DL_load_model():
    new_model = tf.keras.models.load_model('DL_Model_Trained.keras', custom_objects={'smape':smape})
    return new_model

def DL_load_vars():

    with open('DL_vars.pkl', 'rb') as file:
        DL_Model_vars = pickle.load(file)
    X_test, y_test, baseline_score, X_future, X_previous = DL_Model_vars
    return X_test, y_test, baseline_score, X_future, X_previous

def evaluate_DL_model():
    new_model = DL_load_model()
    X_test, y_test, baseline_score, X_future, X_previous = DL_load_vars()
    res = new_model.evaluate(X_test,y_test)
    return res[2], res[4]

def baseline_comparison():
    X_test, y_test, baseline_score, X_future, X_previous = DL_load_vars()
    mae, smape = evaluate_DL_model()
    return ((baseline_score[0] - mae)/baseline_score[0])*100


def predict_last_7days():
    X_test, y_test, baseline_score, X_future, X_previous = DL_load_vars()
    new_model = DL_load_model()
    #print(X_previous.head())
    return new_model.predict(tf.expand_dims(X_previous, 0))


def predict_next_7days():
    X_test, y_test, baseline_score, X_future, X_previous = DL_load_vars()
    new_model = DL_load_model()
    #print(X_future.head())
    return new_model.predict(tf.expand_dims(X_future, 0))


print(type(predict_last_7days()))
print(type(predict_next_7days()))
