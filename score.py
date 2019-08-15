import pickle
import json
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

time_column_name = 'dtime'

def init():
    global model
    model_path = Model.get_model_path(model_name = '_REPLACE_MODEL_NAME_') # the registered model name that we want to deploy
    #model_path = Model.get_model_path(model_name = './model.pkl') # the registered model name that we want to deploy
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

def run(rawdata):
    """
    rawdata is in the following format:
    {'X': X_test.to_json(), 'y': y_test.to_json()}
    """
    try:
        rawobj = json.loads(rawdata) #rawobj is a dict of strings
        X_pred = pd.read_json(rawobj['X'], convert_dates=False)
        X_pred[time_column_name] = pd.to_datetime(X_pred[time_column_name])
        y_pred = np.array(rawobj['y'])
        result = model.forecast(X_pred, y_pred)
    except Exception as e:
        result = str(e)
        return json.dumps({'error': result})

    forecast_as_list = result[0].tolist()
    index_as_df = result[1].index.to_frame().reset_index(drop=True).loc[:,[time_column_name]]
    return json.dumps({'forecast':forecast_as_list, 'index':index_as_df.to_json()})

# for local testing
if __name__ == '__main__':
    init()
    X_test = pd.date_range('2017-08-01', '2017-08-10', freq='D').to_frame(index=False).rename(columns={0:time_column_name})
    y_test = np.full(len(X_test), np.nan, dtype=np.float) 
    test_sample = json.dumps({'X': X_test.to_json(date_format='iso'), 'y' : y_test.tolist()})
    response = run(test_sample)

    res_dict = json.loads(response)
    y_fcst_all = pd.read_json(res_dict['index'])
    y_fcst_all[time_column_name] = pd.to_datetime(y_fcst_all[time_column_name], unit='ms')
    y_fcst_all['forecast'] = res_dict['forecast']
    y_fcst_all.head()