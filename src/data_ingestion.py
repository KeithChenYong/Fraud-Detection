import pandas as pd

def load_df(db_path):
    """Load lung cancer data from SQLite database."""
    df = pd.read_csv(db_path)

    return df
