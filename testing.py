#========================================================
#	IMPORTS
#========================================================

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import main
from main import train_model

import simulator
from simulator import simulator

import yfinance as yf

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler



#========================================================
#	GLOBALS
#========================================================

scaler = MinMaxScaler()

df = yf.download( 'EURPLN=X', end='2030-01-01' )

'''
df = df[ [ 'Close' ] ]
df[ 'FutureClose' ] = df[ 'Close' ].shift( -5 )
df[ 'Direction' ] = np.where( df[ 'FutureClose' ] > df[ 'Close' ], 1, 0 )
df.dropna()
'''

start_date = datetime( 2021, 1, 1 )



#========================================================
#	FUNCTIONS
#========================================================

def get_analyzed_df( df ):

	"""
	:param df:	dataframe to modify;
	"""

	df = df[ ['Close'] ]
	df[ 'FutureClose' ] = df[ 'Close' ].shift( -5 )
	df[ 'Direction' ] = np.where( df['FutureClose'] > df['Close'], 1, 0 )
	df = df.dropna()

	df.index = pd.to_datetime( df.index )

	return df



def add_months( sourcedate, months ):

	"""
	:param sourcedate:	date to add months to,
	:param months:		months to add;
	"""

	month = sourcedate.month - 1 + months
	year = sourcedate.year + month // 12
	month = month % 12 + 1
	day = min( sourcedate.day, 28 )

	return datetime( year, month, day )




def test_gains( df, start_date, steps: int, view_results=True ):

	"""
	:param start_date:	starting date point (turned into datetime),
	:param end_date:	end date point (datetime object),
	:param steps:		how many months ahead to max date to go,
	:param view_results:	final results;
	"""

	ls_final_balances = []
	df = get_analyzed_df( df )
	end_date = df.index.max()

	scaler = MinMaxScaler()

	current_date = start_date

	while ( current_date <= end_date ):

		sub_df = df[ df.index <= current_date ] 	# setting a data range
		test_range_df = df[ current_date: ]

		# model training
		binary_model, history, X_train_class, X_test_class, y_train_class, y_test_class = train_model( sub_df, scaler, 5 )

		# run model simulation
		cycle_gain = simulator( binary_model, test_range_df, 10000.0, 5 )
		ls_final_balances.append( cycle_gain )

		# go to new date
		current_date += relativedelta( months=steps )



	if ( view_results ):
		print( f'All gains: { ls_final_balances }' )

	return None



def test_quarter_forecasting( df, start_date, steps: int, view_results=True ):

	"""
	:param df:		dataframe with other data,
	:oaran start_date:	starting date point (turned into datetime),
        :param steps:           how many months ahead to max date to go,
        :param view_results:    final results;
	"""

	ls_quarter_prices = []
	ls_2_week_forecasts = []

	ls_final_balance = []
	ls_balances = []

	df = get_analyzed_df( df )			# date
	start_train_date = df.index.min()
	end_date = df.index.max()
	current_date = start_date

	close_vals = df[ [ 'Close' ] ].values 		# values


	while ( current_date <= end_date ):
		forecast_date = current_date + relativedelta( months=steps )
		print( f'forecast_date: { forecast_date }' )

		initial_balance = 10000

		for day in range( 14 ):
			train_date = forecast_date + relativedelta( days=day )
			print( f'train_date: { train_date }' )

			train_df = df.loc[ :train_date ]
			binary_model, history, X_train_class, X_test_class, y_train_class, y_test_class = train_model( train_df, scaler, 5 )

			pred_next_day = binary_model.predict( close_vals[ train_date : train_date + timedelta(days=1) ] )

			final_balance = simulator( binary_model, df.iloc[ train_date ], initial_balance, 1 )

			ls_2_week_forecasts.append( pred_next_day )

		ls_quarter_prices.append( ls_2_week_forecasts )
		current_date = forecast_date


	if ( view_results ): print( f'ls_quarter_prices:\n{ ls_quarter_prices }' )

	return ls_quarter_prices



#========================================================
#	MAIN
#========================================================

if ( __name__ == '__main__' ):

	tested_gains = test_quarter_forecasting( df, start_date, 3 )
	print( f"tested gains:\n{ tested_gains }" )
