import argparse
import os
import numpy as np
import pandas as pd
import logging
import azureml.core
from azureml.train.automl import AutoMLConfig
from azureml.core import Workspace, Datastore, Experiment, Dataset
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

print(azureml.core.VERSION)

# Input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--datastore_folder', type=str, default='', dest='datastore_folder', help='folder in datastore where data file is')
parser.add_argument('--file_prefix', type=str, dest='file_prefix', help='actual file is prefix_granularity.csv')
parser.add_argument('--granularity', type=str, dest='granularity', help='forecast granularity, can be hourly, daily, or monthly')
parser.add_argument('--horizon', type=int, dest='horizon', help='how far in the future to forecast, ex. 48 hours, 28 days (4 weeks), 12 months')

# Parse and set parameters
args = parser.parse_args()
datastore_folder = args.datastore_folder
file_prefix = args.file_prefix
granularity = args.granularity
horizon = args.horizon

# Setup AML 
subscription_id = os.environ['AML_SUBSCRIPTION']
resource_group = os.environ['AML_RESOURCE_GROUP']
workspace_name = os.environ['AML_WORKSPACE']
compute_target = os.environ['AML_COMPUTE']
datastore_name = os.environ['AML_DATASTORE']

ws = Workspace(subscription_id, resource_group, workspace_name)
ds = Datastore.get(ws, datastore_name=datastore_name)
compute_target = ws.compute_targets[compute_target]
experiment_name = 'forecast_automl_' + file_prefix + '_' + granularity

# environment for get_data.py
time_column_name = 'dtime'
script_folder = './' # where is get_data.py relative to current folder
script_env = {
    'FORECAST_FILE_PREFIX': file_prefix,
    'FORECAST_GRANULARITY': granularity,
    'FORECAST_HORIZON': horizon
}

# register dataset so get_data can access it
try:
    dataset = Dataset.get(ws, file_prefix)
    print('using existing dataset:{0}'.format(file_prefix))
except:
    data_file = datastore_folder + file_prefix + '_' + granularity + '.csv'
    dataset = Dataset.from_delimited_files(ds.path(data_file))
    dataset = dataset.register(ws, file_prefix)
    print('registered dataset:{0}'.format(file_prefix))

# Setup run configuration
run_config = RunConfiguration(framework="python")
run_config.target = compute_target
run_config.environment.docker.enabled = True
run_config.environment.docker.base_image = azureml.core.runconfig.DEFAULT_CPU_IMAGE
run_config.environment.environment_variables = script_env
dependencies = CondaDependencies.create(pip_packages=["scikit-learn", "scipy", "numpy"])
run_config.environment.python.conda_dependencies = dependencies

# Submit training
automl_config_common = {
    'task': 'forecasting',
    'primary_metric': 'normalized_root_mean_squared_error',
    'verbosity': logging.INFO,
    'time_column_name': time_column_name,
    'max_horizon': horizon,
    'iterations': 10,
    'n_cross_validations': 5,
    'enable_ensembling': True
}

automl_config = AutoMLConfig(path=script_folder,
                             data_script='get_data.py',
                             compute_target=compute_target, 
                             run_configuration=run_config,
                             **automl_config_common)

exp = Experiment(workspace=ws, name=experiment_name)
run = exp.submit(automl_config, show_output=True)
best_run, fitted_model = run.get_output()
