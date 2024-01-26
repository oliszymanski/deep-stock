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
df = df[ [ 'Close' ] ]
df[ 'FutureClose' ] = df[ 'Close' ].shift( -5 )
df[ 'Direction' ] = np.where( df[ 'FutureClose' ] > df[ 'Close' ], 1, 0 )
df.dropna()

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
        :param steps:           how many months ahead to max date to go,
        :param view_results:    final results;
	"""

	forecast_win = 10			# 2 weeks without weekends = 10 days
	ls_final_balances = []

	ls_quarter_prices = []
	ls_2_week_forecasts = []

	end_date = df.index.max()
	current_date = start_date


	while ( current_date <= end_date ):
		forecast_date = current_date + relativedelta( months=steps )
		print( f'forecast_date: { forecast_date }' )

		current_date = forecast_date

	'''
	while ( current_date <= end_date ):
		interval_end_date = current_date + relativedelta( months=steps )

		sub_df = df[ ( df.index >= current_date ) & ( df.index <= interval_end_date ) ]

		forecast_date = current_date
	'''
		# go +3 months
		

		# forecast next day for 2 weeks (10 days):
			# add forecasted next day to ls_2_week_forecasts
			# add ls_2_week_forecasts to ls_quarter_prices
			# reset ls_2_week_forecasts
		# go to the next 3 months


	'''
		while ( forecast_date <= interval_end_date - timedelta( days=forecast_win ) ):
			test_range_df = df[ forecast_date: forecast_date + timedelta( days=forecast_win ) ]

			binary_model, history, X_train_class, X_test_class, y_train_class, y_test_class = train_model( sub_df, scaler, 5 )

			cycle_gain = simulator( binary_model, test_range_df, 1000, 1 )
			ls_final_balances.append( cycle_gain )

			forecast_date += timedelta( days=1 )

		current_date += relativedelta( months=steps )

	'''

	if ( view_results ): print( f'ls_quarter_prices:\n{ ls_quarter_prices }' )

	return ls_quarter_prices



#========================================================
#	MAIN
#========================================================

if ( __name__ == '__main__' ):
	# test_gains( df, start_date, 3 )

	tested_gains = test_quarter_forecasting( df,'2021-01-01', 3 )
	print( f"tested gains:\n{tested_gains}" )
