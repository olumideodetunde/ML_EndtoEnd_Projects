#%%
'''This scripts is used to run the backtesting of the model'''
import warnings
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.feature import main as featuremain
from src.score import calc_fraud_score
warnings.filterwarnings("ignore")

def pred(row: pd.Series, model, features: list) -> float:
    '''This function predicts the fraud score for a given row'''
    x = row[features]
    y_proba = model.predict_proba([x])[:, 1]
    fraud_score = calc_fraud_score(y_proba[0])
    return fraud_score

def main(test:pd.DataFrame, model_path:str, features: list):
    '''Main function to run backtesting'''
    test = featuremain(test)
    test.set_index('transactionTime', inplace=True)
    model = mlflow.sklearn.load_model(model_path)
    test_copy = test.copy()
    threshold = 0.79
    y_pred = (model.predict_proba(test_copy[features])[:, 1] > threshold).astype('float')
    confusion_mattrix = confusion_matrix(test['target'], y_pred)
    y_pred_classes = ['Not Fraud', 'Fraud']
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mattrix,
                                   display_labels=y_pred_classes)
    disp.plot()

if __name__ == "__main__":
    FEATURES = ['availableCash', 'hour', 'day', 'month',
       'average_transaction_amount', 'posEntryMode_0', 'posEntryMode_1','day_or_night',
       'posEntryMode_2', 'posEntryMode_5', 'posEntryMode_7', 'posEntryMode_79',
       'posEntryMode_80', 'posEntryMode_81', 'posEntryMode_90',
       'posEntryMode_91','posEntryMode_91', 'count_1_day', 'count_7_days',
       'count_30_days','count_1_merch_day', 'count_7_merch_days', 'count_30_merch_days']
    test_df = pd.read_csv("data/output/test.csv")
    test_df['posEntryMode_79'] = 0 # adding the missing column
    main(test=test_df, model_path="mlruns/models", features=FEATURES)
    print("Backtesting completed successfully.")
#EOF

