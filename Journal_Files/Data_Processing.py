import numpy as np
import pandas as pd

df_u = pd.read_csv('Data_Xihan_11-5_components/angle_u_1.csv')
print(df_u)

print(df_u.iat[0,0])
print(len(df_u)/8)

for i in range(0, len(df_u)/8):
    if df_u.iat[i,0] == 1:
        print(df_u.iat[i,0])