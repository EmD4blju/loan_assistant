from sklearn.preprocessing import LabelEncoder
import pandas as pd

def clean_data(df):
    #~ Undersampling by loan status
    df_yes = df[df['loan_status'] == 1]
    df_no = df[df['loan_status'] == 0]
    df_no_sampled = df_no.sample(n=len(df_yes))
    df_balanced = pd.concat([df_yes, df_no_sampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df_balanced
    
    #~ Encode categorical variables
    encoder = LabelEncoder()
    df['person_gender'] = encoder.fit_transform(df['person_gender'])
    df['person_education'] = encoder.fit_transform(df['person_education'])
    df['person_home_ownership'] = encoder.fit_transform(df['person_home_ownership'])
    df['loan_intent'] = encoder.fit_transform(df['loan_intent'])
    df['previous_loan_defaults_on_file'] = encoder.fit_transform(df['previous_loan_defaults_on_file'])
    
    #~ Remove anomalies
    Q1 = df['person_age'].quantile(0.25)
    Q3 = df['person_age'].quantile(0.975) # Using 97.5 percentile to retain more data (up to 70 years old)
    IQR = Q3 - Q1
    df = df[(df['person_age'] >= Q1 - 1.5 * IQR) & (df['person_age'] <= Q3 + 1.5 * IQR)]
    return df