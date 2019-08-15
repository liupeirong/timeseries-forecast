import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import calendar
import os
from os import walk
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from azure.storage.blob import BlockBlobService

default_figsize = (20, 5)
default_rotation = 45
default_org_color = 'blue'
default_trans_color = 'red'

def MAPE(actual, pred):
    """
    Calculate mean absolute percentage error.
    Remove NA and values where actual is close to zero
    """
    not_na = ~(np.isnan(actual) | np.isnan(pred))
    not_zero = ~np.isclose(actual, 0.0)
    actual_safe = actual[not_na & not_zero]
    pred_safe = pred[not_na & not_zero]
    APE = 100*np.abs((actual_safe - pred_safe)/actual_safe)
    return np.mean(APE)

def test_stationarity(timeseries, freq):
    """
    Test whether a timeseries has trend
    """
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=freq).mean()
    rolstd = timeseries.rolling(window=freq).std()

    #Plot rolling statistics:
    plt.figure(figsize=default_figsize)
    orig = plt.plot(timeseries, color=default_org_color, label='Original')
    mean = plt.plot(rolmean, color=default_trans_color, label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.xticks(rotation = default_rotation)
    plt.show(block=False)
    plt.close()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
def plot_transformed_data(org, transform, series, trans):
    """
    Plot transformed and original time series data
    """
    plt.figure(figsize=default_figsize)
    plt.plot(org, color = default_org_color, label='Original')
    plt.plot(transform, color = default_trans_color, label = trans)
    plt.legend(loc='best')
    plt.title('%s and %s time-series graph' %(series, trans))
    plt.xticks(rotation = default_rotation)
    plt.show(block=False)
    plt.close()
    

def timeseries_decompose(timeseries, freq):
    """
    Decompose trend, seasonality, and noise
    """
    original = timeseries
    decomposition = seasonal_decompose(timeseries, freq=freq)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.figure(figsize=(20,10))
    plt.subplot(411)
    plt.plot(timeseries, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show(block=False)
    plt.close()
    return decomposition
    
def plot_acf_pacf(timeseries, series):
    """
    Plot acf/pacf to determine ARIMA parameters. 
    Gradual decay on ACF and sharp decay on PACF indicates a AR model.
    Sharp decay on ACF and gradual decay on PACF indicates a MA model.
    It's rare to use both models.
    """
    lag_acf = acf(timeseries)
    lag_pacf = pacf(timeseries, method='ols')

    f, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 5)) 

    #Plot ACF: 
    ax1.plot(lag_acf, color=default_org_color)
    ax1.axhline(y=0,linestyle='--',color='gray')
    ax1.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    ax1.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    ax1.set_title('Autocorrelation Function for %s' %(series))

    #Plot PACF:
    ax2.plot(lag_pacf, color=default_org_color)
    ax2.axhline(y=0,linestyle='--',color='gray')
    ax2.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    ax2.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    ax2.set_title('Partial Autocorrelation Function for %s' %(series))

    plt.tight_layout()
    plt.show(block=False)
    plt.close()

def train_arima(timeseries, series, p, d, q):
    """
    Fit ARIMA on the training set, and examine its accuracy
    """
    # fit ARIMA model on time series
    model = ARIMA(timeseries, order=(p, d, q))  
    results_ = model.fit(disp=-1)  

    # get lengths correct to calculate RSS
    len_results = len(results_.fittedvalues)
    ts_modified = timeseries[-len_results:]

    # calculate root mean square error (RMSE) and mean average percentage error (MAPE)
    rss = sum((results_.fittedvalues - ts_modified)**2)
    rmse = np.sqrt(rss / len(timeseries))
    mape = MAPE(ts_modified, results_.fittedvalues)

    # plot fit
    f, (ax1, ax2) = plt.subplots(2,1, figsize = (20, 10)) 

    ax1.plot(timeseries, color=default_org_color)
    ax1.plot(results_.fittedvalues, color = default_trans_color)
    ax1.set_title('ARIMA model (%i, %i, %i) for ts %s, RMSE: %.4f, MAPE: %.4f' %(p, d, q, series, rmse, mape))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=default_rotation)

    # plot residual errors to see if there's any trend
    residuals = pd.DataFrame(results_.resid)
    ax2.plot(residuals, color=default_org_color)
    ax2.set_title('ARIMA model (%i, %i, %i) for ts %s, residual errors' %(p, d, q, series))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=default_rotation)

    # plot the residual error distribution to see if there's any bias (mean not centered on zero)
    residuals.plot(kind='kde', title='residual distribution')
    print(residuals.describe())

    plt.show(block=False)
    plt.close()

    return results_

def generate_lagged_features(timeseries, series, max_lag, delta):
    for t in range(1, max_lag+1):
        timeseries[series+'_lag'+str(t)] = timeseries[series].shift(t, freq=delta)

def plot_auto_correlation(timeseries, series, freq):
    autocorrelation_plot(timeseries.dropna())
    plt.xlim(0, freq)
    plt.title('Auto-correlation over %s periods for %s'%(freq, series))
    plt.show(block=False)
    plt.close()

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    lastmonthday = calendar.monthrange(year,month)[1]
    return datetime(year, month, lastmonthday)

def rolling_forecast_stationary(train, test, series, order, delta):
    """
    train and test are dataframes
    order is the ARIMA p,i,q
    """
    test_begin, test_end = test.index.min(), test.index.max()
    history = [x for x in train[series]]
    y_forecast = list()
    t = test_begin
    while t <= test_end:
        model = ARIMA(history, order=order)  
        model_fit = model.fit(disp=-1) 
        output = model_fit.forecast()
        yhat = output[0]
        y_forecast.append(yhat)
        history.append(yhat) #put the predictions back to the training set
        obs = test.loc[t]
        print('predicted=%f, expected=%f' % (yhat, obs))
        t = add_months(t, 1) if delta == 'M' else t + delta

    dfforecast = pd.DataFrame(y_forecast, columns={series}, index=test.index)
    mape = MAPE(test, dfforecast)

    return dfforecast,mape

def rolling_forecast_stationary_autoarima(train, test, series, delta, model):
    """
    train and test are dataframes
    """
    test_begin, test_end = test.index.min(), test.index.max()
    history = [x for x in train[series]]
    y_forecast = list()
    t = test_begin
    while t <= test_end:
        model.fit(history)  
        output = model.predict(n_periods=1)
        yhat = output[0]
        y_forecast.append(yhat)
        history.append(yhat)
        obs = test.loc[t]
        print('predicted=%f, expected=%f' % (yhat, obs))
        t = add_months(t, 1) if delta == 'M' else t + delta

    dfforecast = pd.DataFrame(y_forecast, columns={series}, index=test.index)
    mape = MAPE(test, dfforecast)

    return dfforecast,mape

def rolling_forecast_nonstationary(train, test, target_column, logdiff_base, series, order, delta):
    """
    train and test are dataframes
    order is the ARIMA p,i,q
    """
    test_begin, test_end = test.index.min(), test.index.max()
    history = [x for x in train[target_column]] # this contains logdiff
    y_forecast = list()
    t = test_begin
    while t <= test_end:
        model = ARIMA(history, order=order)  
        model_fit = model.fit(disp=-1) 
        output = model_fit.forecast()
        yhat = output[0] # this predicts logdiff
        forecast_log = logdiff_base + yhat
        forecast = np.exp(forecast_log)
        y_forecast.append(forecast)
        history.append(yhat)
        logdiff_base = forecast_log
        obs = test.loc[t]
        print('predicted=%f, expected=%f' % (forecast, obs))
        t = add_months(t, 1) if delta == 'M' else t + delta

    dfforecast = pd.DataFrame(y_forecast, columns={series}, index=test.index)
    mape = MAPE(test, dfforecast)

    return dfforecast,mape

def rolling_forecast_nonstationary_autoarima(train, test, target_column, logdiff_base, series, delta, model):
    """
    train and test are dataframes
    """
    test_begin, test_end = test.index.min(), test.index.max()
    history = [x for x in train[target_column]] # this contains logdiff
    y_forecast = list()
    t = test_begin
    while t <= test_end:
        model.fit(history)  
        output = model.predict(n_periods=1)
        yhat = output[0] # this predicts logdiff
        forecast_log = logdiff_base + yhat
        forecast = np.exp(forecast_log)
        y_forecast.append(forecast)
        history.append(yhat)
        logdiff_base = forecast_log
        obs = test.loc[t]
        print('predicted=%f, expected=%f' % (forecast, obs))
        t = add_months(t, 1) if delta == 'M' else t + delta

    dfforecast = pd.DataFrame(y_forecast, columns={series}, index=test.index)
    mape = MAPE(test, dfforecast)

    return dfforecast,mape

def get_and_evaluate_automl_forecast(X_test, y_test, model, target_column_name):
    """
    X_test is a dataframe with only time column, no target column
    y_test is an array with target values
    returns a dataframe with time as index and forecast as target column, and mape error
    """
    # Replace ALL values in y_test by NaN. 
    # The forecast origin will be at the beginning of the first forecast period
    # (which is the same time as the end of the last training period).
    y_query = y_test.copy().astype(np.float)
    y_query.fill(np.nan)
    # The featurized data, aligned to y, will also be returned.
    # This contains the assumptions that were made in the forecast
    # and helps align the forecast to the original data
    y_predict, X_trans = model.forecast(X_test, y_query)

    mape = MAPE(y_test, y_predict)
    dfforecast = pd.DataFrame(index = X_test.index)
    dfforecast[target_column_name] = y_predict
    return dfforecast, mape

def multi_series_wide_to_tall(data, time_column_name, series_column_name, value_column_name):
    """
    input dataframe schema: index (number), time_column, target_columns
    series_column_name: the result column that identifies a series (group)
    value_column_name: the result column that contains the time series data value
    """
    dfunpivot = pd.melt(data, id_vars=[time_column_name], value_vars=data.columns.drop(time_column_name), \
                        var_name=series_column_name, value_name=value_column_name)
    dfunpivot.index = dfunpivot[time_column_name]
    dfunpivot.drop([time_column_name], axis=1, inplace=True)
    return dfunpivot

def clean_data(df, time_column_name, target_column_name, output_folder):
    """
    input dataframe is hourly based
    input dataframe schema: index (number), time_column, target_column
    output: 
        cleaned dataframe
        a report of data quality
        hourly, daily, monthly data stored to output_folder
    
    cleaning is done by:
    * Remove rows with duplicate timestamps
    * Fill missing timestamps
    * Remove outliers (negatives, zeros, values beyond 3 stdev)
    * Fill missing target values by interpolation 
        using 2 closest non-missing values with a limit of max 6 consequtive missing values
        fill the remaining null with mean
    """
    report = {}
    report['name']=target_column_name
    # Remove rows before the first non-na and after the last non-na
    dfnotna = df[df[target_column_name].notna()]
    if len(dfnotna) <= 0:
        report['num_rows'] = 0
        return dfnotna, report
    report['num_rows'] = len(dfnotna)
    data_begin = dfnotna.iloc[0][time_column_name]
    data_end = dfnotna.iloc[-1][time_column_name]
    df = df[(df[time_column_name] >= data_begin) & (df[time_column_name] <= data_end)]
    # Remove duplicate timeslots
    duplicate_time_rows = df[df[time_column_name].duplicated()].count()[0]
    report['duplicate_time_rows'] = duplicate_time_rows
    if duplicate_time_rows > 0:
        df = df[~df[time_column_name].duplicated(keep='first')]
    # Build full range of timestamps, reindex to fill the gaps
    min_time, max_time = df[time_column_name].min(), df[time_column_name].max()
    report['min_time'], report['max_time'] = min_time, max_time
    dt_idx = pd.date_range(min_time, max_time, freq='H')
    df.index = pd.DatetimeIndex(df[time_column_name])
    df = df.reindex(dt_idx)  
    df.drop(time_column_name, axis=1, inplace=True)
    df_missing = df[df.isnull().all(axis=1)]
    missing_timeslot_count = len(df_missing)
    report['missing_timeslot_count'] = missing_timeslot_count
    # Remove outliers
    target_mean = df[target_column_name].mean()
    target_stdev = df[target_column_name].std()
    outlier = df.loc[(np.abs(df[target_column_name]-target_mean) > (3*target_stdev)) & ~(df[target_column_name].isnull()), target_column_name]
    df.loc[outlier.index, target_column_name] = np.nan
    zeroneg = df.loc[df[target_column_name] <= 0, target_column_name]
    df.loc[zeroneg.index, target_column_name] = np.nan
    outlier_count = outlier.count()
    zeroneg_count = zeroneg.count()
    report['outlier_count'], report['zeroneg_count'] = outlier_count, zeroneg_count
    # Fill missing values
    df = df.interpolate(limit=6, method='linear')
    null_row_count_after_interpolation = df[target_column_name].isnull().sum()
    report['null_row_count_after_interpolation'] = null_row_count_after_interpolation
    if null_row_count_after_interpolation > 0:
        df.fillna(target_mean, inplace=True)
    # Aggregate data to daily and monthly
    dfday = df[[target_column_name]].groupby(pd.Grouper(freq=timedelta(days=1))).agg({target_column_name: 'sum'})
    dfmonth = df[[target_column_name]].groupby(pd.Grouper(freq='M')).agg({target_column_name: 'sum'})
    
    # store hourly, daily, monthly data
    df.to_csv(os.path.join(output_folder, target_column_name + '_hourly.csv'), index_label=time_column_name, float_format='%.3f')
    dfday.to_csv(os.path.join(output_folder, target_column_name + '_daily.csv'), index_label=time_column_name, float_format='%.3f')
    dfmonth.to_csv(os.path.join(output_folder, target_column_name + '_monthly.csv'), index_label=time_column_name, float_format='%.3f')
        
    return df, report

def download_from_blob(local_folder, blob_account, blob_key, blob_container, blob_prefix):
    if not os.path.exists(local_folder):
        os.makedirs(local_folder, exist_ok=True)
        
    block_blob_service = BlockBlobService(account_name=blob_account, account_key=blob_key)
    container_name = blob_container
    blobs = list(block_blob_service.list_blobs(container_name, prefix=blob_prefix))
    for blob in blobs:
        print('downloading {}'.format(blob.name))
        block_blob_service.get_blob_to_path(container_name, blob.name, os.path.join(local_folder, blob.name))

def upload_to_blob(local_folder, file_prefix, blob_account, blob_key, blob_container, blob_folder):
    block_blob_service = BlockBlobService(account_name=blob_account, account_key=blob_key)
    
    for (dirpath, dirnames, filenames) in walk(local_folder):
        for file_name in filenames:
            if file_name.startswith(file_prefix):
                file_path = os.path.join(dirpath, file_name)
                blob_name = os.path.join(blob_folder, file_name)
                print('uploading {} to {}'.format(file_path, blob_name))
                block_blob_service.create_blob_from_path(blob_container, blob_name, file_path)
    