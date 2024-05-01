# import sys
import os
import configparser
from data_ingestion import load_df
from data_processing import original_df
from model import train_model
from evaluate import evaluate_model

def load_config(config_path):
    """Load configurations from the specified config file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def main():
    ''' Main python script to execute loading of data, propressing, model, cross validation, evaluating'''
    # Load data
    current_dir = os.getcwd()

    # # Define the relative path to cancer_patient_data_sets.db based on the current directory
    if "src" in current_dir:
        db_path = "../data/PS_20174392719_1491204439457_log.csv"
        config_path = "../config.ini"
    else:
        db_path = "data/PS_20174392719_1491204439457_log.csv"
        config_path = "config.ini"       

    df = load_df(db_path)

    # Explore, clean & preprocess + Dimension reduction & feature engineering
    training_set, test_set = original_df(df)

    # Load configurations from the config file
    config = load_config(config_path)
    
    # Train and evaluate models
    model = train_model(training_set, config)
    evaluate_model(model, test_set)

if __name__ == "__main__":
    main()