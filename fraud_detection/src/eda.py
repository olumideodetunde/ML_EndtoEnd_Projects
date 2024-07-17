'''Exploratory script'''
import pandas as pd
train = pd.read_csv("data/output/train.csv")
test = pd.read_csv("data/output/test.csv")
data_df = pd.concat([train, test])
data_df

#How many fraudulent transactions are there as a percentage?
print(data_df['target'].value_counts())
print(data_df['target'].value_counts(normalize=True) * 100)

#How much has been lost due to fraud?
print(f"Total fraud loss - £{data_df[data_df['target'] == 1]['transactionAmount'].sum()}")

# How many accounts have fraudulent transactions?
x = data_df[data_df['target'] == 1].groupby('accountNumber')['transactionAmount']
print(f"Total number of accounts with fraudulent transactions - {x.count().shape[0]}")
