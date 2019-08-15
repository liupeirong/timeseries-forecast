import numpy as np
import pandas as pd
import os
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--input_data_folder', type=str, default='data', dest='input_data_folder', help='local folder of the data csv')
parser.add_argument('--input_file_name', type=str, default='nyc_energy.csv', dest='input_file_name', help='input csv file')
parser.add_argument('--output_data_folder', type=str, default='output', dest='output_data_folder', help='local folder for processed output')

args = parser.parse_args()
time_column_name = 'dtime'
target_column_name = 'nycdemand'

input_data_folder = args.input_data_folder
output_data_folder = args.output_data_folder
input_file_name = args.input_file_name

datafile = os.path.join(input_data_folder, input_file_name)
df = pd.read_csv(datafile, header=0, low_memory=False)
print("data shape: ", df.shape)

# delete rows with null timestamps
df.dropna(subset=[time_column_name], inplace=True)

os.makedirs(output_data_folder, exist_ok=True)
dfclean, report = utils.clean_data(df[[time_column_name, target_column_name]], \
                  time_column_name, target_column_name, output_data_folder)

dfreport = pd.DataFrame([report])
dfreport.to_csv(os.path.join(output_data_folder, 'data_processing_report.csv'), index=False)
