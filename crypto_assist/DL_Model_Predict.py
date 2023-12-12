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
    X_test, y_test, baseline_score = DL_Model_vars

    return X_test, y_test, baseline_score

X_test, y_test, baseline_score = DL_load_vars()

def evaluate_DL_model():

    res = new_model.evaluate(X_test,y_test)
    return res[2], res[4]

def baseline_comparison():
    mae, smape = evaluate_DL_model()
    return ((baseline_score[0] - mae)/baseline_score[0])*100


def predict_last_7days():
    return new_model.predict(X_test)[-1]

def actual_last_7days():
    return y_test[-1]


print(evaluate_DL_model())
print(baseline_comparison())
print(predict_last_7days())
print(actual_last_7days())
