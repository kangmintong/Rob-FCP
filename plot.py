# import plotly.graph_objects as go
#
# import pandas as pd
# from datetime import datetime
#
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
#
# fig = go.Figure(data=[go.Candlestick(x=df['Date'],
#                 open=df['AAPL.Open'],
#                 high=df['AAPL.High'],
#                 low=df['AAPL.Low'],
#                 close=df['AAPL.Close'])])
#
# fig.show()

import plotly.graph_objects as go
from datetime import datetime
import numpy as np

eps = 0.001
data = np.array([0.84, 0.9])
data2 = np.array([0.84-eps, 0.9+eps])
std = np.array([0.02, 0.02])
x = [' ', '  ']

fig = go.Figure(data=[go.Candlestick(x=x,
                       open=data, high=data+std,
                       low=data+std*-1, close=data2)])

fig.show()