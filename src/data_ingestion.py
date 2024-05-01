import pandas as pd
import os

def load_df(db_path):
    """Load lung cancer data from csv"""

    if os.path.exists(db_path):
        df = pd.read_csv(db_path)
    else:
        print(f"Error: Database file '{db_path}' not found. Please check your data file path.")
        return  # Exit the function or handle the error as needed

    return df
