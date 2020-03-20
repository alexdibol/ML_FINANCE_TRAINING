import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
import yfinance
from pandas_datareader import data as pdr
import fix_yahoo_finance

def create_dataframe():
	df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
	return df2
	
def create_plot():
	t = np.arange(0.0, 2.0, 0.01)
	s = 1 + np.sin(2 * np.pi * t)

	fig, ax = plt.subplots()
	ax.plot(t, s)

	ax.set(xlabel='time (s)', ylabel='voltage (mV)',
        title='About as simple as it gets, folks')
	ax.grid()
	plt.show()

def load_info():
	datas = pdr.get_data_yahoo('AAPL',start=datetime.datetime(2006, 10, 1), end=datetime.datetime(2012, 1, 1))	
	return datas

def load_series(stock, start, end):
	datos = pdr.get_data_yahoo(stock,start,end)	
	return datos

def create_graph(df):
	fig=plt.figure(figsize=(12,10))
	plt.plot(df['Close'])
	plt.grid()
	plt.title("CLOSING PRICE")
	plt.show()