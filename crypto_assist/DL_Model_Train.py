import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from keras import models
from keras import layers
from keras import optimizers, metrics
from keras.layers.experimental.preprocessing import Normalization
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from keras.layers import Lambda
from keras.optimizers.schedules import ExponentialDecay
from keras import regularizers
from keras import Model
import os
import pickle

from crypto_assist.data_preprocess import preprocessor_not_scaled

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


def train_model():


    def preprocessing():
        df = preprocessor_not_scaled('BTCUSDT_daily.csv','final_text_df.csv','social_number_data.csv','FearAndGreedIndex.csv')
        df_new = df.drop(columns=['date','volume BTC','twitter_favourites','reddit_active_users','twitter_followers','RSI_class','ADX_class'])
        return df_new


    N_FEATURES = preprocessing().shape[1]  # All features other than time
    N_TARGETS = 1                          # Prediciting only 1 target (close price)
    TARGET = 'close'

    FOLD_LENGTH = int(len(preprocessing())/2)     # Assume fold length of half the dataset
    FOLD_STRIDE = int(len(preprocessing())/4)     # Assume stride every 10 days
    TRAIN_TEST_RATIO = 0.5                        # 50-50 split ratio

    def get_folds(df: pd.DataFrame,
        fold_length: int,
        fold_stride: int) -> list[pd.DataFrame]:

        folds = []
        for index in range(0, fold_length, fold_stride):
            if index + fold_length > len(df):
                break
            fold = df.iloc[index:index + fold_length +1,:]
            folds.append(fold)
        return folds


    folds = (get_folds(preprocessing(),FOLD_LENGTH,FOLD_STRIDE))
    print(f'Number of folds created were {len(folds)}')
    print(f'Each with a shape equal to {folds[0].shape}.')

    fold = folds[-1]  #Using most recent Fold

    INPUT_LENGTH = 10            # We can assume 10 days for a forecating period
    OUTPUT_LENGTH = 5            # If we want predict 5 ahead
    TEMP_TRAIN_TEST_RATIO = 0.95 # How we want to split each fold (can be same as train test ratio)



    def temporal_train_test_split(fold:pd.DataFrame,
                                temp_train_test_ratio: float,
                                input_length: int) -> tuple[pd.DataFrame]:
        # Train set
        last_train_index = round(temp_train_test_ratio * len(fold))
        fold_train = fold.iloc[0:last_train_index, :]

        #Test Set
        first_test_index = last_train_index - input_length
        fold_test = fold.iloc[first_test_index:, :]

        return (fold_train,fold_test)

    (fold_train, fold_test) = temporal_train_test_split(fold, TEMP_TRAIN_TEST_RATIO, INPUT_LENGTH)


    print(fold_train.shape)
    print(fold_test.shape)


    STRIDE = 1  #How many days we want to go through the fold

    def get_Xi_yi_7(first_index: int,
                fold: pd.DataFrame,
                input_length: int,
                output_length: int) -> tuple[np.ndarray, np.ndarray]:
        '''
        - extracts one sequence from a fold
        - returns a pair (Xi, yi) with:
            * len(Xi) = `input_length` and Xi starting at first_index
            * len(yi) = `output_length`
            * last_Xi and first_yi separated by the gap = horizon -1
        '''

        Xi_start = first_index
        Xi_last = Xi_start + input_length
        yi_start = Xi_last
        yi_last = yi_start + output_length

        Xi = fold[Xi_start:Xi_last]
        yi = fold[yi_start:yi_last][TARGET]

        return (Xi, yi)


    def get_X_y_7(fold: pd.DataFrame,
                input_length: int,
                output_length: int,
                stride: int,
                shuffle=True) -> tuple[np.ndarray, np.ndarray]:
        """
        - Uses `data`, a 2D-array with axis=0 for timesteps, and axis=1 for (targets+covariates columns)
        - Returns a Tuple (X,y) of two ndarrays :
            * X.shape = (n_samples, input_length, n_covariates)
            * y.shape =
                (n_samples, output_length, n_targets) if all 3-dimensions are of size > 1
                (n_samples, output_length) if n_targets == 1
                (n_samples, n_targets) if output_length == 1
                (n_samples, ) if both n_targets and lenghts == 1
        - You can shuffle the pairs (Xi,yi) of your fold
        """
        X = []
        y = []

        # Scanning the fold/data entirely with a certain stride
        for i in range(0, len(fold), stride):
            ## Extracting a sequence starting at index_i
            Xi, yi = get_Xi_yi_7(first_index=i,
                                fold=fold,
                                input_length=input_length,
                                output_length=output_length)
            ## Exits loop as soon as we reach the end of the dataset
            if len(yi) < output_length:
                break
            X.append(Xi)
            y.append(yi)

        X = np.array(X)
        y = np.array(y)
        y = np.squeeze(y)


        if shuffle:
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X = X[idx]
            y = y[idx]

        return X, y



    X_train, y_train = get_X_y_7(fold=fold_train,
                            input_length=INPUT_LENGTH,
                            output_length=OUTPUT_LENGTH,
                            stride=STRIDE)
    X_test, y_test = get_X_y_7(fold=fold_test,
                            input_length=INPUT_LENGTH,
                            output_length=OUTPUT_LENGTH,
                            stride=STRIDE)

    print("Shapes for the training set:")
    print(f"X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")

    print("Shapes for the test set:")
    print(f"X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

    def last_seen_value_baseline(X, y):

        # How many values do you want to predict in the future ?
        output_length = y.shape[-1]

        # For each sequence, let's consider the last seen value
        # and only the close column
        last_seen_values = X[:,-1, 3].reshape(-1,1)

        # We need to duplicate these values as many times as output_length
        repeated = np.repeat(last_seen_values, axis = 1, repeats = output_length)

        MAE = np.mean(np.abs(y - repeated))
        MAPE = np.mean(abs((y - repeated) / y)) * 100
        SMAPE = np.mean((abs(y - repeated)) / ((y + repeated)/2))* 100
        return MAE, MAPE, SMAPE


    baseline_score = last_seen_value_baseline(X_test, y_test)
    print(baseline_score)


    X_future = fold[:][-INPUT_LENGTH:]
    X_previous = fold[:][-(INPUT_LENGTH+OUTPUT_LENGTH):-OUTPUT_LENGTH]
    print (X_future.index)
    print (X_previous.index)


    def smape(y_true, y_pred):
        return tf.reduce_mean(((abs(y_true - y_pred)) / ((y_true + y_pred)/2))* 100, axis=-1)


    def init_model(X_train, y_train):

        # adam = optimizers.Adam(learning_rate=0.02)


        initial_learning_rate = 0.001 # Default Adam

        # lr_schedule = ExponentialDecay(
        # initial_learning_rate,
        # decay_steps = 10000,    # every 500 iterations
        # decay_rate = 0.9)      # we multiply the learning rate by the decay_rate


        # adam = optimizers.Adam(learning_rate=lr_schedule)
        adam = optimizers.Adam()


        reg_l2 = regularizers.L2(0.05)

        output_length = y_train.shape[-1]

        normalizer = Normalization(input_shape=[INPUT_LENGTH,N_FEATURES])

        normalizer.adapt(X_train)

        model = models.Sequential()

        model.add(normalizer)

        model.add(layers.LSTM(400,
                            activation='tanh',
                            return_sequences = True
                            ))
        model.add(layers.LSTM(100,
                            activation='tanh',
                            return_sequences = True
                            ))
        # model.add(layers.LSTM(128,
        #                       activation='tanh',
        #                       return_sequences = True, #kernel_regularizer = reg_l2
        #                       ))
        # model.add(layers.LSTM(64,
        #                       activation='tanh',
        #                       return_sequences = True, #kernel_regularizer = reg_l2
        #                       ))
        model.add(layers.LSTM(50,
                            activation='tanh',
                            return_sequences = False, #kernel_regularizer = reg_l2
                            ))
        # model.add(layers.Dropout(0.3))
        model.add(layers.Dense(100, activation='relu'
                            ))
        # model.add(layers.Dropout(0.3))
        model.add(layers.Dense(50, activation='relu'
                            ))
        model.add(layers.Dense(output_length, activation='linear'))

        model.compile(loss= 'mse', optimizer=adam, metrics=["mse",'mae','mape',smape])

        return model


    def fit_model(model: tf.keras.Model, verbose=1) -> tuple[tf.keras.Model, dict]:

        es = EarlyStopping(monitor = "val_loss",
                        patience = 150,
                        mode = "min",
                        restore_best_weights = True)

        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                            #   patience=30, min_lr=0.000001)


        history = model.fit(X_train, y_train,
                            validation_split = 0.2,
                            shuffle = False,
                            batch_size = 64,
                            epochs = 1000,
                            callbacks = [es],
                            verbose = verbose)

        return model, history



    model, history = fit_model(init_model(X_train, y_train))
    print(type(model))
    model.save('DL_Model_Trained.keras')

    return X_test, y_test, baseline_score, X_future, X_previous


DL_Model_vars = train_model()

with open('DL_vars.pkl', 'wb') as file:
    pickle.dump(DL_Model_vars, file)
