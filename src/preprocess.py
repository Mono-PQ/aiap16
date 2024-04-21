import pandas as pd 
import numpy as np
import sqlite3
import requests
from sklearn.preprocessing import StandardScaler

def download_db(url, file_path):
    """Function to query database from specified url in db_url.
    Args:
        url (string): website where .db is hosted 
        file_path (string): file path where .db file is saved
    """
    response = requests.get(url)
    response.raise_for_status() 
    
    with open(file_path, 'wb') as file:
        file.write(response.content)

def query_db(file_path, query):
    """Function to query database from specified url in db_url.
    Args:
        file_path (string): file path where .db file is saved
        query (string): SQL query to retrieve full data of lung_cancer table
    Returns:
        df (pandas.DataFrame): data from lung_cancer db table
    """
    conn = sqlite3.connect(file_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def save_data(df, path):
    """Function to save data to specified path
    Args:
        df (pandas.DataFrame): raw data queried from online database
        path (string): file path where raw data will be saved
    """
    df.to_csv(path, index=False)

def calculate_length_smoking(row):
    if len(row['Start Smoking']) == 4 and len(row['Stop Smoking']) == 4:
        return int(row['Stop Smoking']) - int(row['Start Smoking'])
    elif row['Stop Smoking'] == "Still Smoking" and len(row['Start Smoking']) == 4 :
        return 2024 - int(row['Start Smoking'])
    else:
        return np.nan

def preprocess_data(df):
    """Preprocessing required to prepare the data for modelling
    Args:
        df (pandas.DataFrame): data retrieved from lung_cancer db
    Returns:
        df (pandas.DataFrame): cleaned numerical data after preprocessing
    """
    df_cleaned = df.drop_duplicates(keep='first')
    df_cleaned = df_cleaned[(df_cleaned['Age'] <= 115) & (df_cleaned['Age'] >= 0)] 
    df_cleaned['Gender'] = df_cleaned['Gender'].str.title() 
    df_cleaned = df_cleaned.query("Gender != 'Nan'") 
    df_cleaned = df_cleaned.dropna(subset=['Air Pollution Exposure']) 
    df_cleaned = df_cleaned.fillna("Unknown") 
    df_cleaned['Dominant Hand'] = df_cleaned['Dominant Hand'].apply(lambda x: "Both" if x == "RightBoth" else x)
    return df_cleaned

def feature_engineer(df):
    """Add features required for modelling
    Args:
        df (pandas.DataFrame): preprocessed data from lung_cancer db
    Returns:
        df (pandas.DataFrame): augment dataframe with additional features and removal of columns where data were augmented
    """
    df["Weight changed"] = df["Current Weight"] - df['Last Weight']
    df["Is Smoker"] = df["Start Smoking"].apply(lambda x: 0 if x == "Not Applicable" else 1)
    df['Smoking Length'] = df.apply(calculate_length_smoking, axis=1)
    df['Smoking Length'] = df['Smoking Length'].fillna(0)
    df = df.drop(['Last Weight', 'Start Smoking', 'Stop Smoking'], axis=1)
    return df

def standard_scaler(df):
    """Scaling of numerical features for modelling
    Args:
        df (pandas.DataFrame): cleaned data after preprocessing
    Returns:
        df (pandas.DataFrame): data with numerical columns scaled using StandardScaler
    """
    numeric_cols_to_scale = ['Age', 'Current Weight', 'Weight changed', 'Smoking Length']
    scaler = StandardScaler()
    df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
    return df

def encoder(df):
    """Map categorical features to numerical values
    Args:
        df (pandas.DataFrame): cleaned data after preprocessing
    Returns:
        df (pandas.DataFrame): data with categorical columns mapped to numerical value for modelling 
    """
    gender_map = {'Male': 0, 'Female': 1}
    yesno_map = {'Unknown': 0, 'No': 1, 'Yes': 2}
    genetic_map = {'Not Present': 0, 'Present': 1}
    airpollution_map = {'Low': 0, 'Medium': 1, 'High': 2}
    tiredness_map = {'None / Low': 0, 'Medium': 1, 'High': 2}
    dominanthand_map = {'Both': 0, 'Right': 1, 'Left': 2}

    df['Gender'] = df['Gender'].map(gender_map)
    df['COPD History'] = df['COPD History'].map(yesno_map)
    df['Genetic Markers'] = df['Genetic Markers'].map(genetic_map)
    df['Air Pollution Exposure'] = df['Air Pollution Exposure'].map(airpollution_map)
    df['Taken Bronchodilators'] = df['Taken Bronchodilators'].map(yesno_map)
    df['Frequency of Tiredness'] = df['Frequency of Tiredness'].map(tiredness_map)
    df['Dominant Hand'] = df['Dominant Hand'].map(dominanthand_map)
    return df