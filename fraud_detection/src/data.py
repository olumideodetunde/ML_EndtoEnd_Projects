'''This script is the data preprocessing script for the project'''
import pandas as pd

def create_target(label_file:str) -> pd.DataFrame:
    '''This function creates the target variable for the label dataframe'''
    labels_df = pd.read_csv(label_file)
    labels_df['target'] = 1
    return labels_df

def convert_to_datetime(df:pd.DataFrame, col:str) -> pd.DataFrame:
    '''This function converts a column to datetime'''
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def create_dataset(data_file:str, label:pd.DataFrame,
                   join_col:str) -> pd.DataFrame:
    '''This function creates the dataset and fills target with 
    0 for non-frauds'''
    data_df = pd.read_csv(data_file)
    data_df = pd.merge(data_df, label, on=join_col, how="left")
    data_df['target'] = data_df['target'].fillna(0)
    return data_df

def get_holdout_set(df:pd.DataFrame, date_col:str,
                    duration:int=1) -> pd.DataFrame:
    '''This function splits the data into train and test set using
    a cutoff duration in months'''
    df = df.sort_values(by=date_col, ascending=True)
    cutoff = df[date_col].max() - pd.DateOffset(months=duration)
    df.set_index(date_col, inplace=True)
    train = df[df.index <= cutoff]
    test = df[df.index > cutoff]
    return train, test

if __name__ == "__main__":
    LABEL = 'data/raw/labels_obf.csv'
    DATA = 'data/raw/transactions_obf.csv'
    LABEL_DF = create_target(LABEL)
    LABEL_DF = convert_to_datetime(df=LABEL_DF, col='reportedTime')
    DATA_DF = create_dataset(data_file=DATA, label=LABEL_DF, join_col='eventId')
    DATA_DF = convert_to_datetime(df=DATA_DF, col='transactionTime')
    TRAIN, TEST = get_holdout_set(df=DATA_DF, date_col='transactionTime', duration=2)
    TRAIN.to_csv("data/output/train.csv")
    TEST.to_csv("data/output/test.csv")
    print("Data Preprocessing complete")
