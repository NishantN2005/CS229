import numpy as np
import pandas as pd


data=pd.read_excel('Real estate valuation data set.xlsx')
data_np=np.array(data)

print(data_np)

