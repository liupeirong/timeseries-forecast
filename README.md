# timeseries-forecast
Time series forecasting using ARIMA and Azure Machine Learning.

## Set up
* Use a Azure Data Science VM or any VM in Azure to set up a dev environment as documented [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment). 
* In the Azure ML Conda environment, also install:
    * matplotlib
    * azure-storage-blob

## Sample data
We use a public dataset of NYC energy demand data on hourly basis between 2012 and 2017. The dataset csv is copied from [here](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/automated-machine-learning/forecasting-energy-demand). 
