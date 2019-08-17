import argparse
import os
import pandas as pd
import numpy as np
import json
import azureml.core
from azureml.core import Workspace
from azureml.core.webservice import Webservice

print(azureml.core.VERSION)

# Input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--file_prefix', type=str, dest='file_prefix', help='actual file is prefix_granularity.csv')
parser.add_argument('--granularity', type=str, dest='granularity', help='forecast granularity, can be hourly, daily, or monthly')
parser.add_argument('--from_datetime', type=str, dest='from_datetime', help='forecast from datetime')
parser.add_argument('--horizon', type=int, dest='horizon', help='how much in the future to forecast, unit depends on granularity, can be hour, day, or month')

# Parse and set parameters
args = parser.parse_args()
file_prefix = args.file_prefix
granularity = args.granularity
from_datetime = args.from_datetime
horizon = args.horizon

# Setup AML 
subscription_id = os.environ['AML_SUBSCRIPTION']
resource_group = os.environ['AML_RESOURCE_GROUP']
workspace_name = os.environ['AML_WORKSPACE']

ws = Workspace(subscription_id, resource_group, workspace_name)
experiment_name = 'forecast_automl_' + file_prefix + '_' + granularity
model_name = experiment_name.replace('-', '').replace('_', '').lower()

# Prepare input for forecasting
time_column_name = 'dtime'
freq = granularity[0].upper()
X_test = pd.date_range(start=from_datetime, periods=horizon, freq=freq).to_frame(index=False).rename(columns={0:time_column_name})
y_test = np.full(len(X_test), np.nan, dtype=np.float) 
test_sample = json.dumps({'X': X_test.to_json(date_format='iso'), 'y' : y_test.tolist()})
print('input json:{}'.format(test_sample))

# Find the web service 
service = Webservice(ws, model_name)

# Call the web service
response = service.run(test_sample)
print('output json:{}'.format(response))

# Parse results 
res_dict = json.loads(response)
y_fcst_all = pd.read_json(res_dict['index'])
y_fcst_all[time_column_name] = pd.to_datetime(y_fcst_all[time_column_name], unit='ms')
y_fcst_all['forecast'] = res_dict['forecast']
y_fcst_all.index = y_fcst_all[time_column_name]
y_fcst_all.drop(time_column_name, axis=1, inplace=True)
y_fcst_all.sort_index(inplace=True)
print(y_fcst_all)
