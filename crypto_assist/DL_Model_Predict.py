import tensorflow as tf
from keras import models
import pickle
#from crypto_assist.DL_Model_Train import smape


def smape(y_true, y_pred):
        return tf.reduce_mean(((abs(y_true - y_pred)) / ((y_true + y_pred)/2))* 100, axis=-1)

new_model = tf.keras.models.load_model('DL_Model_Trained.keras', custom_objects={'smape':smape})


def DL_load_vars():

    with open('DL_vars.pkl', 'rb') as file:
        DL_Model_vars = pickle.load(file)
    X_test, y_test, baseline_score, X_future, X_previous = DL_Model_vars

    return X_test, y_test, baseline_score, X_future, X_previous

X_test, y_test, baseline_score, X_future, X_previous = DL_load_vars()

def evaluate_DL_model():

    res = new_model.evaluate(X_test,y_test)
    return res[2], res[4]

def baseline_comparison():
    mae, smape = evaluate_DL_model()
    return ((baseline_score[0] - mae)/baseline_score[0])*100


def predict_last_7days():
    #print(X_previous.head())
    return new_model.predict(tf.expand_dims(X_previous, 0))


def predict_next_7days():
    #print(X_future.head())
    return new_model.predict(tf.expand_dims(X_future, 0))


print(evaluate_DL_model())
print(baseline_comparison())
print(predict_last_7days())
print(predict_next_7days())
