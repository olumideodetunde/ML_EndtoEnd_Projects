#%%
import mlflow
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import average_precision_score, classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from src.feature import split_on_time
from src.feature import main as feature_main

def prepare_mldata(data:str, features:list) -> pd.DataFrame:
    '''This function prepares the data for training the model.'''
    ml_df = pd.read_csv(data)
    ml_df = feature_main(ml_df)
    ml_df = ml_df[features]
    return ml_df

def upsample_datapoints(df:pd.DataFrame,
                        target:pd.DataFrame) -> pd.DataFrame:
    '''This function upsamples the minority class in the dataset.'''
    df = df.join(target)
    majority = df[df.target==0]
    minority = df[df.target==1]
    minority_upsampled = resample(minority, 
                                replace=True,     
                                n_samples=majority.shape[0], 
                                random_state=123)
    df = pd.concat([majority, minority_upsampled])
    return df

def main(data:str, features:list):
    '''This is the training function for the model.'''
    # ml_df = prepare_mldata(data, features) ---------------------------------> this leaks data
    # x_train, x_val, y_train, y_val = split_on_time(ml_df,
    #                                 'transactionTime')
    # train = upsample_datapoints(x_train, y_train) #upsample the train data only
    # x_train, y_train = train.drop('target', axis=1), train['target']

    # Load the full dataset
    ml_df = pd.read_csv(data)
    train_df, val_df = split_on_time(ml_df, 'transactionTime')
    x_train = prepare_mldata(train_df, features)
    y_train = train_df['target']
    x_val = prepare_mldata(val_df, features)
    y_val = val_df['target']
    
    
    with mlflow.start_run(): 
        xgb = GradientBoostingClassifier()
        xgb.fit(x_train, y_train)
        y_prob = xgb.predict_proba(x_val)[:,1]
        y_pred = xgb.predict(x_val)
        aucpr = average_precision_score(y_pred, y_prob)
        precision = precision_score(y_pred, y_val)
        recall = recall_score(y_pred, y_val)
        mlflow.log_metric("aucpr", aucpr)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.save_model(xgb, "mlruns/models")

if __name__ == "__main__":
    FEATURES = ['transactionTime','availableCash',  'target', 'hour', 'day', 'month',
       'average_transaction_amount', 'posEntryMode_0', 'posEntryMode_1','day_or_night',
       'posEntryMode_2', 'posEntryMode_5', 'posEntryMode_7', 'posEntryMode_79',
       'posEntryMode_80', 'posEntryMode_81', 'posEntryMode_90',
       'posEntryMode_91','posEntryMode_91', 'count_1_day', 'count_7_days',
       'count_30_days','count_1_merch_day', 'count_7_merch_days',
       'count_30_merch_days']
    main("data/output/train.csv", FEATURES)
    print("Model trained and exported successfully.")
#EOF
