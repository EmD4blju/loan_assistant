import os.path

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def histplot(series: pd.Series, title: str, xlabel:str,ylabel:str) ->None:
    plt.figure(figsize=(10, 6))
    sns.histplot(series, bins=15, kde=True)
    plt.title(title)
    plt.axvline(series.mean(), color='r', linestyle='--', label='Mean')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(Path("neural_core","analisys",title))
    plt.clf()

def countplot(series: pd.Series, title: str, xlabel:str,ylabel:str) ->None:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=series)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.savefig(Path("neural_core","analisys",title))
    plt.clf()

analisys_dir=Path("neural_core","analisys")
df = pd.read_csv(Path("neural_core","repo", "loan_data.csv"))
if not os.path.exists(analisys_dir):
    os.mkdir(analisys_dir)
histplot(df['loan_amnt'],'Distribution of Loan Amounts','Loan Amount','Frequency')
countplot(df['loan_intent'],'Loan Intent Distribution','Loan Intent','Count')
histplot(df['loan_int_rate'],'Distribution of Loan Interest Rates','Loan Interest Rate (%)','Frequency')
histplot(df['loan_percent_income'],'Distribution of DTI (Debt-to-Income Ratio)','Loan Interest Rate (%)','Frequency')
histplot(df['credit_score'],'Distribution of Credit Scores','Credit Score','Frequency')
countplot(df['previous_loan_defaults_on_file'],'Distribution of Previous Loan Defaults on File','Previous Loan Defaults on File','Frequency')
countplot(df['loan_status'],'Loan Status Distribution','Loan Status','Count')
images =[]
for name in os.listdir(analisys_dir):
    full_path=Path.joinpath(analisys_dir,name)
    img = cv2.imread(full_path)
    images.append(img)
    os.remove(full_path)
cv2.imwrite(Path.joinpath(analisys_dir,'analisys.png'),cv2.vconcat(images))

