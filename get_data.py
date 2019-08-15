import argparse
import os
import pandas as pd
import numpy as np
from datetime import timedelta
import azureml.core
from azureml.core import Workspace, Run, Dataset

def split_train_test_by_granularity(granularity, horizon, min_time, max_time):
    hourly, daily, monthly = 'hourly', 'daily', 'monthly'
    if (granularity == hourly):
        delta = timedelta(hours=1)
        lookback = 90 * 24 # 90 days, how many data points do we look back for training
        frequency = 24 # data is hourly, we examine stationarity, seasonality, trend etc on daily basis
        # if total data is less than 2x horizon, abort
        total_hours = (max_time-min_time).total_seconds()/3600
        if (total_hours) < (2 * horizon): 
            print('{} hours are not enough to train a model to forecast {} hours'.format(total_hours, horizon))
            return 
        # leave the last horizon for testing
        test_slice_end = max_time
        test_slice_begin = test_slice_end - timedelta(hours=horizon-1)
        training_slice_end = test_slice_begin - timedelta(hours=1) 
        training_slice_begin = training_slice_end - timedelta(hours=lookback-1)
    elif (granularity == daily):
        delta = timedelta(days=1)
        lookback = 730 # 2 years, how many data points do we look back for training
        frequency = 7 # data is daily, we examine stationarity, seasonality, trend etc on weekly basis
        # if total data is less than 2x horizon, abort
        total_days = (max_time-min_time).total_seconds()/3600/24
        if (total_days) < (2 * horizon): 
            print('{} days are not enough to train a model to forecast {} days'.format(total_days, horizon))
            return 
        # leave the last horizon for testing
        test_slice_end = max_time
        test_slice_begin = test_slice_end - timedelta(days=horizon-1)
        training_slice_end = test_slice_begin - timedelta(days=1) 
        training_slice_begin = training_slice_end - timedelta(days=lookback-1)
    elif (granularity == monthly):
        delta = 'M'
        lookback = 120 # 10 years, how many data points do we look back for training
        frequency = 12 # data is monthly, we examine stationarity, seasonality, trend etc on yearly basis
        # if total data is less than 2x horizon, abort
        total_months = (max_time-min_time).total_seconds()/3600/24/30
        if (total_months) < (2 * horizon): 
            print('{} months are not enough to train a model to forecast {} months'.format(total_months, horizon))
            return 
        # timedelta doesn't support month, leave the last year for testing
        test_slice_end = max_time
        test_slice_begin = test_slice_end - timedelta(days=364)
        training_slice_end = test_slice_begin - timedelta(days=1) 
        training_slice_begin = min_time
    else:
        raise Exception('Unknown granularity {}. Valid values are {},{},{}'.format(granularity, hourly, daily, monthly))

    return delta, frequency, training_slice_begin, training_slice_end, test_slice_begin, test_slice_end

def compute_train_test_for_automl(data, training_begin, training_end, test_begin, test_end, target_column_name, time_column_name):
    """
    Input:
    data: is a dataframe with time in the index, and target columns, no time_column
    Returns:
    X_train, X_test: dataframes without target values
    y_train, y_test: target values in arrays
    """
    X_train = data.loc[training_begin:training_end, ]
    X_test = data.loc[test_begin:test_end, ]
    # AutoML takes arrays
    y_train = X_train.pop(target_column_name).values  #this also removes target values from the original dataframe
    y_test = X_test.pop(target_column_name).values
    # AutoML needs time column
    X_train[time_column_name] = X_train.index
    X_test[time_column_name] = X_test.index
    
    return X_train, X_test, y_train, y_test

def get_data():
    time_column_name = 'dtime'
    
    target_column_name = os.environ['FORECAST_FILE_PREFIX']
    granularity = os.environ['FORECAST_GRANULARITY'] 
    horizon = int(os.environ['FORECAST_HORIZON'])

    print('target:{}, granularity:{}, horizon:{}' \
        .format(target_column_name, granularity, horizon))

    # get data from dataset, below is equivalent to the following local call
    #df = pd.read_csv(csvfile, header=0, index_col=0, parse_dates=True)
    run = Run.get_context()
    workspace = run.experiment.workspace
    dataset = Dataset.get(workspace, target_column_name)
    df = dataset.to_pandas_dataframe()
    df.index = df[time_column_name]
    df.drop(time_column_name, inplace = True, axis=1)
    
    min_time, max_time = df.index.min(), df.index.max()
    try_split = split_train_test_by_granularity(granularity, horizon, min_time, max_time)
    if try_split is None:
        raise Exception('can not train data', min_time, max_time)
    else:
        (delta, frequency, training_slice_begin, training_slice_end, test_slice_begin, test_slice_end) = try_split
        print('train between %s and %s, forecast between %s and %s' % \
        (training_slice_begin, training_slice_end, test_slice_begin, test_slice_end))

    df = df.loc[training_slice_begin:test_slice_end, ]
    X_train, X_test, y_train, y_test = compute_train_test_for_automl(df, \
        training_slice_begin, training_slice_end, test_slice_begin, test_slice_end, target_column_name, time_column_name)

    return X_train, y_train
