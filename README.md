# Timeseries Forecasting
This sample explores different methods for timeseries forecasting, including
* Statistical algorithms - __ARIMA__ and __Auto ARIMA__ 
* Machine Learning - __Random Forest__ and __Azure AutoML__
* Off-the-shelf solutions - __Azure Data Explorer__ and __Power BI__  

It also demonstrates how to use Azure Machine Learning Service to train, register, and deploy a forecasting model as a web service. 

We use the hourly NYC energy demand dataset between 2012 and 2017. The dataset csv is copied from [here](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/automated-machine-learning/forecasting-energy-demand). 

## Set up
* Use a Azure Data Science VM or any VM in Azure to set up a dev environment as documented [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment). 
* In the Azure ML Conda environment, in addition to the basic AzureML packages, also install the following packages:
  * matplotlib
  * azure-storage-blob
  * pyramid-arima
  * azureml-sdk[explain,automl]

## Explore and clean the data
Use the [process_data](notebooks/process_data.ipynb) Jupyter notebook to explore the data. This notebook illustrates how to fill the time series with missing timeslots, remove outliers, and aggregate the data to handle the respective seasonality for different forecasting granularity -
* hourly patterns repeated daily
* daily patterns repeated weekly
* monthly patterns repeated yearly 

## ARIMA
Use the [arima](notebooks/arima.ipynb) Juputer notebook to explore how to test stationarity, and if data is not stationary, how to remove trend and seasonality to forecast on the residual and then add trend and seasonality back in the forecast. 

Determining the parameters for ARIMA requires a lot of trial and error, even with the help of ACF (auto correlation function) and PACF (partial auto correlation function) graphs. Auto ARIMA tries different parameters automatically and often produces much better results with far less effort. It's also not necessary to make the data stationary for Auto ARIMA.

## Machine Learning
With machine learning, we transform the data out of the timeseries domain into, for example, regression problems. It's not necessary to convert data to stationary for machine learning. The [machine_learning](notebooks/machine_learning.ipynb) Jupyter notebook explores Random Forest for forecasting by manually adding features such as lags and day of week. The sample dataset does include weather data which is often very helpful in this type of forecasting. We didn't use weather data in this case because we want to mimic datasets that don't have weather data available.

Azure AutoML forecasting is capable of fitting different ML models and choosing the best model with stack or voting ensemble. It's also not necessary to manually calcuate the lags for AutoML.  

## Putting it all together
Once you are happy exploring the data and models locally, you can use python scripts to operationalize the machine learning pipeline. 
* [01_process_data.py](01_process_data.py) - cleans and aggregates data for different forecasting granularity
* [02_submit_training.py](02_submit_training.py) - trains Azure AutoML forecasting models in Azure ML compute. Azure ML automatically tracks all the experiment runs and scales compute resources when running training jobs 
* [03_register_and_deploy.py](03_register_and_deploy.py) - once you are happy with a model, register it with Azure ML, and use Azure ML to automatically create a web service for forecasting
* [04_forecast_from_webservice.py](04_forecast_from_webservice.py) - demonstrates how to call the deployed web service for forecasting

TODO:
* release pipeline for training
* release pipeline for deployment 

## Off-the-shelve solutions
[Azure Data Explorer](https://docs.microsoft.com/en-us/azure/data-explorer/time-series-analysis) has powerful built-in time series analysis capabilities. [timeseries.kql](AzureDataExplorer/timeseries.kql) contains sample scripts that analyze the sample dataset for forecasting. For example, below chart is the output of Azure Data Explorer decomposing the sample data:
![Alt text](/AzureDataExplorer/decomposition.png?raw=true "Azure Data Explorer decomposition") 

Power BI also has built-in capabilities for time series forecasting. The [sample pbix file](PowerBI/nyc_forecast_daily.pbix) demonstrates how to use Power BI to forecast on the sample data:
![Alt text](/PowerBI/forecasting.png?raw=true "Power BI forecasting")

## Other considerations
### Multiple series - single model or multiple models
If you have multiple series of data, most likely you'll need to train a model for each series. This could lead to hundreds or thousands of models which could be difficult to maintain.  Consider grouping them based on similarity to reduce the number of models.

If you treat different series as different grains (for example, energy demand for power plant A and plant B), training just one model could possibly work if A and B share the same time range, granularity, scale of target value, and other features such as weather.

### Other algorithms 
The following statistical, machine learning, and deep learning models, although not demonstrated in this example, have also proven effective in time series forecasting use cases. 
* [Holt Winters](https://otexts.com/fpp2/holt-winters.html)
* [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
* [Facebook Prophet](https://facebook.github.io/prophet/)
* [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)
