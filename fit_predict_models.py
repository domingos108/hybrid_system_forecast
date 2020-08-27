import numpy as np
import pandas as pd
from sklearn import preprocessing

import  utils  as ut


def fit_sklearn_model(ts, model, test_size, val_size):
    train_size = len(ts) - test_size - val_size
    y_train = ts['actual'][0:train_size]
    x_train = ts.drop(columns=['actual'], axis=1)[0:train_size]

    return model.fit(x_train, y_train)


def predict_sklearn_model(ts, model):

    x = ts.drop(columns=['actual'], axis=1)
    return model.predict(x)

def additive_hybrid_model(predicted, real, time_window, base_model, test_size, val_size,
                          result_options, title, type_data):
    P = predicted
    train_size = len(predicted) - test_size

    ts_atu = real

    errors = np.subtract(real, P)

    # normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(errors[0:train_size].reshape(-1, 1))
    normalized_error = min_max_scaler.transform(errors.reshape(-1, 1))

    # fit_predict

    error_values = pd.DataFrame({'actual': normalized_error.flatten()})

    error_windowed = ut.create_windowing( df=error_values, lag_size=time_window)

    pi = fit_sklearn_model(ts=error_windowed, model=base_model,
                               test_size=test_size, val_size=val_size)

    pi_pred = predict_sklearn_model(ts=error_windowed, model=pi)
    # _____________________________

    pi_pred = min_max_scaler.inverse_transform(pi_pred.reshape(-1, 1)).flatten()

    P = P[time_window:] + pi_pred

    ts_atu = ts_atu[time_window:]

    return ut.make_metrics_avaliation(ts_atu, P, test_size,
                                       val_size,
                                       result_options, base_model.get_params(deep=True),
                                       title + '(tw' + str(time_window) + ')')


def nolic_model(linear_forecast, real, nonlinear_forecast, time_window, 
                base_model, test_size, val_size,
                title, result_options, type_data):

    train_size_represents = len(real) - test_size
    
    error_values = nonlinear_forecast - linear_forecast

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(real[0:train_size_represents].reshape(-1, 1))

    min_max_scaler_linear = preprocessing.MinMaxScaler()
    min_max_scaler_linear.fit(linear_forecast[0:train_size_represents].reshape(-1, 1))

    min_max_scaler_error = preprocessing.MinMaxScaler()
    min_max_scaler_error.fit(error_values[0:train_size_represents].reshape(-1, 1))

    real_normalized = min_max_scaler.transform(real.reshape(-1, 1)).flatten()
    linear_normalized = min_max_scaler_linear.transform(linear_forecast.reshape(-1, 1)).flatten()
    error_normalized = min_max_scaler_error.transform(error_values.reshape(-1, 1)).flatten()
      

    tsf_part = ut.create_windowing(lag_size=(time_window - 1),
                                    df=pd.DataFrame({'actual': linear_normalized}))

    ef_part = ut.create_windowing(lag_size=(time_window - 1),
                                   df=pd.DataFrame({'actual': error_normalized}))

    real_part = ut.create_windowing(lag_size=(time_window - 1),
                                     df=pd.DataFrame({'actual': real_normalized}))

    tsf_part.columns = ['ts_prev' + str(i) for i in reversed(range(0, time_window))]
    ef_part.columns = ['error_prev' + str(i) for i in reversed(range(0, time_window))]

    ts_formated = pd.concat([ef_part, tsf_part,
                             real_part['actual']], axis=1)

    p = fit_sklearn_model(ts=ts_formated, model=base_model,
                              test_size=test_size, val_size=val_size)

    pred = predict_sklearn_model(ts=ts_formated,
                                     model=p)

    pred = min_max_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    real_atu = real[(time_window - 1):]
    
    result_metrics = ut.make_metrics_avaliation(real_atu, pred, test_size,
                                                val_size,
                                                result_options, base_model.get_params(deep=True),
                                                title + '(tw' + str(time_window) + ')', None)
    return result_metrics