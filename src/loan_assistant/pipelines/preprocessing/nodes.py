import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    #~ Undersampling by loan status
    df_yes = df[df['loan_status'] == 1]
    df_no = df[df['loan_status'] == 0]
    df_no_sampled = df_no.sample(n=len(df_yes))
    df_balanced = pd.concat([df_yes, df_no_sampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df_balanced
    #~ Remove anomalies
    Q1 = df['person_age'].quantile(0.25)
    Q3 = df['person_age'].quantile(0.975) # Using 97.5 percentile to retain more data (up to 70 years old)
    IQR = Q3 - Q1
    df = df[(df['person_age'] >= Q1 - 1.5 * IQR) & (df['person_age'] <= Q3 + 1.5 * IQR)].copy()
    
    #~ Remove colinearity
    df = df.drop(columns=['person_gender', 'person_age', 'cb_person_cred_hist_length'])
    return df


def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders = {}
    categorical_features = ['person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
    for feature in categorical_features:
        encoder = LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature])
        encoders[feature] = encoder
    return df, encoders
    

def scale_numerical(df: pd.DataFrame, scaler: TransformerMixin) -> Tuple[pd.DataFrame, TransformerMixin]:
    numeric_features = ['person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'person_emp_exp', 'credit_score']
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df, scaler


def split_data(df: pd.DataFrame, val_size: float=0.2, test_size: float=0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train, X_test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['loan_status'])
    X_train, X_val = train_test_split(X_train, test_size=val_size, random_state=42, stratify=X_train['loan_status'])
    return X_train, X_val, X_test