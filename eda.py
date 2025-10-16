import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _classify_education(level):
    level = str(level).lower()
    if "high school" == level:
        return 0
    elif "associate" == level:
        return 1
    elif "bachelor" == level:
        return 2
    elif "master" == level:
        return 3
    elif "doctorate" == level:
        return 4
    else:
        return np.nan
    
def _classify_home_ownership(status):
    status = str(status).lower()
    if status in ["rent", "other"]:
        return 0
    elif status == "mortgage":
        return 1
    elif status == "own":
        return 2
    else:
        return np.nan
    
class EDAProcessor:
    """
    Load CSV, run EDA preprocessing and feature engineering, and return cleaned/filtered DataFrame.
    Usage:
      proc = EDAProcessor("loan_data.csv")
      filtered = proc.load_and_process()
    """
    def __init__(self, csv_path="loan_data.csv"):
        self.csv_path = csv_path
        self.gender_le = None
        self.load_defaults_le = None
        self.scaler = None
    
    def load_and_process(self):
        df = pd.read_csv(self.csv_path)

        num_cols = df.select_dtypes(include=['float64','int64']).columns
        print("Numerical columns:", num_cols.tolist())
        cat_cols = df.select_dtypes(include=['object']).columns

        # Data preprocessing
        gender_le = LabelEncoder()
        load_defaults_le = LabelEncoder()
        df["person_gender"] = gender_le.fit_transform(df["person_gender"].astype(str))
        df["previous_loan_defaults_on_file"] = load_defaults_le.fit_transform(df["previous_loan_defaults_on_file"].astype(str))
        self.gender_le = gender_le
        self.load_defaults_le = load_defaults_le
        df = pd.get_dummies(df, columns=["loan_intent"], drop_first=True)

        if len(num_cols) > 0:
            num_cols = num_cols.drop("loan_status")
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            self.scaler = scaler

        # Feature engineering
        df["person_education"] = df["person_education"].apply(_classify_education)
        df["person_home_ownership"] = df["person_home_ownership"].apply(_classify_home_ownership)

        return df

if __name__ == "__main__":
    proc = EDAProcessor("loan_data.csv")
    processed_df = proc.load_and_process()
    print(processed_df.head(10))