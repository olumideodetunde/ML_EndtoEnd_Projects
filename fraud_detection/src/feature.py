'''This is the feature engineering module for the project'''
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder

def get_hour_day_month_year(df:pd.DataFrame, date_col:str):
    '''This function returns the dataframe with the hour, day, month and year of the transaction'''
    df[date_col] = pd.to_datetime(df[date_col])
    df['hour'] = df[date_col].dt.hour
    df['day'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    return df

def get_average_transaction_amount(df:pd.DataFrame,
                                   group_by_cols:list):
    '''This function return the average transaction amount 
    for each account number'''
    df['average_transaction_amount'] = df.groupby(group_by_cols)\
        ['transactionAmount'].transform('mean')
    return df

def encode_payment_method(df: pd.DataFrame, column: str) -> pd.DataFrame:
    '''This function encodes the payment methods'''
    df[column] = df[column].astype(str)
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[[column]])
    cat_col = encoder.get_feature_names_out([column])
    df = pd.concat([df, pd.DataFrame(encoded, columns=cat_col)], axis=1)
    return df

def encode_day_and_night(df: pd.DataFrame, column: str) -> pd.DataFrame:
    '''This function encodes the day and night in the hour column'''
    df[column] = df[column].astype(str)
    df['day_or_night'] = df['hour'].apply(lambda x: 1 if x in range(6, 18) else 0)
    return df

def rolling_transaction_counts(df, time_col: str):
    """
    This function adds rolling transaction counts for 1 day, 
    7 days, and 30 days to the dataframe.
    """
    df = df.sort_values(by=time_col)
    periods = {'1d': 'count_1_day', '7d': 'count_7_days',
               '30d': 'count_30_days'}
    for period, col_name in periods.items():
        temp = pd.Series(df.index, index=df[time_col], name=col_name)
        count = temp.rolling(period).count() - 1
        count.index = temp.values
        df[col_name] = count.reindex(df.index)
    return df

def rolling_merchant_counts(df, time_col: str):
    """
    This function adds rolling transaction counts for 1 day, 
    7 days, and 30 days to the dataframe.
    """
    df = df.sort_values(by=time_col)
    periods = {'1d': 'count_1_merch_day', '7d': 'count_7_merch_days',
               '30d': 'count_30_merch_days'}
    for period, col_name in periods.items():
        temp = pd.Series(df.index, index=df[time_col], name=col_name)
        count = temp.rolling(period).count() - 1
        count.index = temp.values
        df[col_name] = count.reindex(df.index)
    return df

def split_on_time(df:pd.DataFrame, date_col:str) -> pd.DataFrame:
    '''This function gives a temporal split into train and val set'''
    df = df.sort_values(by=date_col, ascending=True)
    df.set_index(date_col, inplace=True)
    time = TimeSeriesSplit(n_splits=3)
    x, y = df.drop('target', axis=1), df['target']
    for train_index, test_index in time.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return x_train, x_test, y_train, y_test

def main(df:pd.DataFrame) -> pd.DataFrame:
    '''This function returns the dataframe with all the features'''
    df = get_hour_day_month_year(df, 'transactionTime')
    df = encode_day_and_night(df, 'hour')
    df = get_average_transaction_amount(df, ['accountNumber'])
    df = encode_payment_method(df, 'posEntryMode')
    df = df.groupby('accountNumber').apply(lambda x:
        rolling_transaction_counts(x, 'transactionTime')).reset_index(drop=True)
    df = df.groupby('merchantId').apply(lambda x:
        rolling_merchant_counts(x, 'transactionTime')).reset_index(drop=True)
    return df

if __name__ == "__main__":
    data = pd.read_csv("data/output/train.csv")
    data = main(data)
    print("Feature engineering completed successfully.")
#EOF
