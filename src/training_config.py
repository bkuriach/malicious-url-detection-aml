from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.runconfig import DockerConfiguration
from azureml.widgets import RunDetails
from azureml.core import Workspace
from azureml.core import Dataset
import azureml.core

ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))
ws.set_default_datastore('maliciousurldatastore')
default_ds = ws.get_default_datastore()
env = Environment.from_conda_specification("experiment_env", "environment.yml")
malicious_ds = ws.datasets.get("malicious-url-dataset")

script_config = ScriptRunConfig(source_directory='src',
                              script='train.py',
                              arguments = ['--regularization', 0.1, # Regularizaton rate parameter
                                           '--input-data', malicious_ds.as_named_input('training_data')], # Reference to dataset
                              environment=env,
                              docker_runtime_config=DockerConfiguration(use_docker=True)) 

# submit the experiment
experiment_name = 'malicious-url'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)
run.wait_for_completion()