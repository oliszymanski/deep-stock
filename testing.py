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
#	FUNCTIONS
#========================================================

def get_analyzed_df( df ):

	df = df[ ['Close'] ]
	df[ 'FutureClose' ] = df[ 'Close' ].shift( -5 )
	df[ 'Direction' ] = np.where( df['FutureClose'] > df['Close'], 1, 0 )
	df = df.dropna()

	df.index = pd.to_datetime( df.index )

	return df



def add_months( sourcedate, months ):
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



#========================================================
#	MAIN
#========================================================

df = yf.download( 'EURPLN=X', end='2030-01-01' )
scaler = MinMaxScaler()

start_date = datetime( 2021, 1, 1 )


if ( __name__ == '__main__' ):
	test_gains( df, start_date, 3 )
