import pandas as pd

df = pd.read_csv("featured_data.csv")

grouped_df = df.groupby('symbol')['date'].agg(min_date = 'min', max_date = 'max').reset_index()
grouped_df.columns = ['symbol', 'min_date', 'max_date']
print(grouped_df)

