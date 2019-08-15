import argparse
import os
import azureml.core
from azureml.core import Workspace, Experiment, Model
from azureml.train.automl.run import AutoMLRun
from azureml.core.conda_dependencies import CondaDependencies

print(azureml.core.VERSION)

# Input parameters
parser = argparse.ArgumentParser()
parser.add_argument('--file_prefix', type=str, dest='file_prefix', help='actual file is prefix_granularity.csv')
parser.add_argument('--granularity', type=str, dest='granularity', help='forecast granularity, can be hourly, daily, or monthly')

# Parse and set parameters
args = parser.parse_args()
file_prefix = args.file_prefix
granularity = args.granularity

# Setup AML 
subscription_id = os.environ['AML_SUBSCRIPTION']
resource_group = os.environ['AML_RESOURCE_GROUP']
workspace_name = os.environ['AML_WORKSPACE']

ws = Workspace(subscription_id, resource_group, workspace_name)
experiment_name = 'forecast_automl_' + file_prefix + '_' + granularity

# Register the model from last best run
print('registering the latest model for {0}'.format(experiment_name))
exp = Experiment(workspace=ws, name=experiment_name)
run_generator = exp.get_runs()
run_latest = next(run_generator)
if run_latest.get_status() != 'Completed' or run_latest.type != 'automl':
    raise Exception('the last run is not completed or is not automl')

run_id = run_latest.get_details()['runId']
automl_run = AutoMLRun(exp, run_id)
best_run, fitted_model = automl_run.get_output()
model_name = experiment_name.replace('-', '').replace('_', '').lower()
# Register a model
model = best_run.register_model(model_name=model_name, model_path='outputs/model.pkl')
# Get existing model
#model=Model(ws, model_name)

# Figure out the run's dependencies
conda_env_file_name = '{}_env.yml'.format(experiment_name)
localenv = CondaDependencies.create(conda_packages=['numpy','scikit-learn'], pip_packages=['azureml-sdk[automl]'])
localenv.save_to_file('.', conda_env_file_name)
best_iteration = int(str.split(best_run.id,'_')[-1])      # the iteration number is a postfix of the run ID.
dependencies = automl_run.get_run_sdk_dependencies(iteration = best_iteration)
for p in ['azureml-core']:
    print('{}\t{}'.format(p, dependencies[p]))
with open(conda_env_file_name, 'r') as cefr:
    content = cefr.read()
with open(conda_env_file_name, 'w') as cefw:
    cefw.write(content.replace(azureml.core.VERSION, dependencies['azureml-core']))

# Replace the model name in score.py
script_file_name = 'score.py'
with open(script_file_name, 'r') as cefr:
    content = cefr.read()
with open(script_file_name, 'w') as cefw:
    cefw.write(content.replace('_REPLACE_MODEL_NAME_', model_name))

# Create a container image
from azureml.core.image import Image, ContainerImage
image_config = ContainerImage.image_configuration(runtime= "python",
                                 execution_script = script_file_name,
                                 conda_file = conda_env_file_name,
                                 tags = {'type': "automl-forecasting"},
                                 description = experiment_name)
# Create a new container image
image = Image.create(name = model_name,
                     models = [model],
                     image_config = image_config, 
                     workspace = ws)
image.wait_for_creation(show_output = True)
if image.creation_state == 'Failed':
    raise Exception("Image build log at: " + image.image_build_log_uri)
# Get an existing image
#image = Image(ws, name=model_name)

# Deploy the docker image to ACI 
from azureml.core.webservice import AciWebservice
aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 2, 
                                               tags = {'type': "automl-forecasting"},
                                               description = experiment_name)
from azureml.core.webservice import Webservice
aci_service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                            image = image,
                                            name = model_name,
                                            workspace = ws)
aci_service.wait_for_deployment(True)
print(aci_service.state)
