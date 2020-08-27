import datetime
import pickle as pkl

import pandas as pd
import numpy as np


class result_options:
    test_result = 0
    val_result = 1
    train_result = 2
    save_result = 3


def mean_square_error(y_true, y_pred):
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)

    return np.square(np.subtract(y_true, y_pred)).mean()

def root_mean_square_error(y_true, y_pred):

    return mean_square_error(y_true, y_pred)**0.5


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(np.where(y_true == 0)[0]) > 0:
        return np.inf
    else:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.mean(np.abs((y_true - y_pred) / (( np.abs(y_true) + np.abs(y_pred) )/2) )) * 100

def mean_absolute_error(y_true, y_pred):
    
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.mean(np.abs(y_true - y_pred))


def u_theil(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred[0:(len(y_pred) - 1)], y_pred[1:(len(y_pred))])).sum()

    return error_sup / error_inf


def average_relative_variance(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred, mean)).sum()

    return error_sup / error_inf


def index_agreement(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.abs(np.subtract(y_true, y_pred))).sum()

    error_inf = np.abs(np.subtract(y_pred, mean)) + np.abs(np.subtract(y_true, mean))
    error_inf = np.square(error_inf).sum()

    return 1 - (error_sup / error_inf)


def prediction_of_change_in_direction(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    true_sub = np.subtract(y_true[0:(len(y_true) - 1)], y_true[1:(len(y_true))])
    pred_sub = np.subtract(y_pred[0:(len(y_pred) - 1)], y_pred[1:(len(y_pred))])

    mult = true_sub * pred_sub
    result = 0
    for m in mult:
        if m > 0:
            result = result + 1

    return (100 * (result / len(y_true)))

def gerenerate_metric_results(y_true, y_pred):

    return {'MSE': mean_square_error(y_true, y_pred),
            'RMSE':root_mean_square_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'SMAPE':symmetric_mean_absolute_percentage_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'theil': u_theil(y_true, y_pred),
            'ARV': average_relative_variance(y_true, y_pred),
            'IA': index_agreement(y_true, y_pred),
            'POCID': prediction_of_change_in_direction(y_true, y_pred),
            }

def make_metrics_avaliation(y_true, y_pred, test_size,
                            val_size,return_type,model_params,
                            title, prevs_df=None):
    data_size = len(y_true)
    train_size = data_size - (val_size + test_size)

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    y_true_test = y_true[(data_size - test_size):data_size]
    y_pred_test = y_pred[(data_size - test_size):data_size]

    val_result = None

    if val_size>0:
        y_true_val = y_true[(train_size):(data_size - test_size)]
        y_pred_val = y_pred[(train_size):(data_size - test_size)]
        val_result = gerenerate_metric_results(y_true_val, y_pred_val)

    y_true_train = y_true[:train_size]
    y_pred_train = y_pred[:train_size]

    geral_dict = {
        'test_metrics': gerenerate_metric_results(y_true_test, y_pred_test),
        'val_metrics': val_result,
        'train_metrics': gerenerate_metric_results(y_true_train, y_pred_train),
        'real_values': y_true,
        'predicted_values': y_pred,
        'pool_prevs': prevs_df,
        'params': model_params
    }

    if return_type == 0:
        return geral_dict['test_metrics']
    elif return_type == 1:
        return geral_dict['val_metrics']
    elif return_type == 2:
        return geral_dict['train_metrics']
    elif return_type == 3:
        return save_result(geral_dict, title)

def save_result(dict_result, title):

    currentDT = datetime.datetime.now()
    title = title+"-"+currentDT.strftime('%d%m%y%s')+".pkl"
    
    with open(title, 'wb') as handle:
        pkl.dump(dict_result, handle)

    #print("exported to pkl")
    return title

def open_saved_result(file_name):
    with open(file_name, 'rb') as handle:
        b = pkl.load(handle)
    return b

def create_windowing(df, lag_size):
    final_df = None
    for i in range(0, (lag_size + 1)):
        serie = df.shift(i)
        if (i == 0):
            serie.columns = ['actual']
        else:
            serie.columns = [str('lag' + str(i))]
        final_df = pd.concat([serie, final_df], axis=1)

    return final_df.dropna()

def do_computations(data, path, metric, metrics_to_use):
    
    best = None
    metric_values = pd.DataFrame() 

    execs = 30
    if len(path) == 1:
        
        best = open_saved_result(path[0])
        best_path = path[0]
        for i in range(0,execs):
            metric_values =metric_values.append(pd.DataFrame([best['test_metrics']])) 
            
            
    elif len(path) == execs:
        for i2 in path:
            
            result_model= open_saved_result(i2)     
            metric_values = metric_values.append(pd.DataFrame([result_model['test_metrics']])) 
            
            if best==None:
                best= result_model
                best_path = i2
            else:
                if best[metrics_to_use][metric]>=result_model[metrics_to_use][metric]:
                    best= result_model
                    best_path = i2

    result = pd.DataFrame()
    
    result['MSE'] = [best['test_metrics']['MSE']]
    result['MAPE'] = [best['test_metrics']['MAPE']]  
    result['ARV'] = [best['test_metrics']['ARV']]
    result['theil'] = [best['test_metrics']['theil']]
    result['MAE'] = [best['test_metrics']['MAE']]
    #result['POCID'] = [best['test_metrics']['POCID']]
    return result, metric_values, best