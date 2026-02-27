import pandas as pd

file_path = 'cs_trade_events_hold3_rsi_zle-1_n10.csv'

df = pd.read_csv(file_path)



file_path_2 = 'featured_data.csv'

df_2 = pd.read_csv(file_path_2)


print(df[(df['symbol'] == 'AAPL') & (df['date'] == '2016-07-25')])

print(df_2[(df_2['symbol'] == 'AAPL') & (df_2['date'] == '2016-07-25')])
