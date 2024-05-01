import sys
import os
import configparser
from data_ingestion import load_df
from data_processing import original_df
from model import train_model
from evaluate import evaluate_model

from kaggle.api.kaggle_api_extended import KaggleApi
def download_kaggle_dataset(dataset_path, download_path='./'):
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset_path, path=download_path, unzip=True)

def load_config(config_path):
    """Load configurations from the specified config file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def main():
    ''' Main python script to execute loading of data, propressing, model, cross validation, evaluating'''
    # Kaggle path
    kaggle_dataset_path = 'ealaxi/paysim1'
    download_path = '../data'

    # Download the Kaggle dataset
    download_kaggle_dataset(kaggle_dataset_path, download_path)

    # Load data from the downloaded dataset
    data_path = os.path.join(download_path, 'PS_20174392719_1491204439457_log.csv')
    if not os.path.exists(data_path):
        print(f"Error: Dataset file '{data_path}' not found.")
        return

    df = load_df(data_path)

    # Define the relative path based on the current directory
    current_dir = os.getcwd()
    if "src" in current_dir:
        config_path = "../config.ini"
    else:
        config_path = "config.ini"       
    # Load configurations from the config file
    config = load_config(config_path)

    # Explore, clean & preprocess + Dimension reduction & feature engineering
    training_set, test_set = original_df(df)
    
    # Train and evaluate models
    model = train_model(training_set, config)
    evaluate_model(model, test_set)

if __name__ == "__main__":
    main()