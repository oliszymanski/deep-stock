#==========================================================
#	IMPORTS
#===========================================================

import yfinance as yf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model

import main



#===========================================================
#	GLOBALS
#===========================================================

initial_balance = 1000.0
look_ahead = 5

df = yf.download( 'EURPLN=X', end='2090-01-01' )
df = df[ [ 'Close' ] ]
df[ 'FutureClose' ] = df[ 'Close' ].shift( -5 )
df[ 'Direction' ] = np.where( df[ 'FutureClose' ] > df[ 'Close' ], 1, 0 )
df.dropna()

df.index = pd.to_datetime( df.index )
end_date = df.index.max()
start_date = end_date - pd.DateOffset( months=24 )		# start predicting from the data 2 years before
df = df.loc[ start_date : end_date ]



#===========================================================
#	FUNCTIONS
#===========================================================

def get_column_data( df, val_00 : str , val_01 : str ):
	return df[ [ val_00 ] ].values, df[ val_01 ]



def simulator( model, df, initial_balance : float, look_ahead : int, view_results=False ):

	"""
	:param model:			binary classification model,
	:param df:			test data dataframe,
	:param initial_balance:		initial balance to simulate with,
	:param look_ahead:		steps to look ahead during predicition;
	"""

	X, y = get_column_data( df, 'Close', 'Direction' )
	future_prices = []

	cash_balance = initial_balance
	foreign_currency_balance = 0

	for i in range( len(X) - look_ahead ):
		current_data = X[ i : i+1 ]
		future_price_pred = model.predict( current_data.reshape( 1, -1, 1 ) )
		future_prices.append( future_price_pred )


	for i in range( look_ahead, len(X) ):
		current_price = X[ i, 0 ]
		future_price = future_prices[ i - look_ahead ]
		prediction = y[i]

		if ( prediction == 1 ):		# if price is going to go high
			buy_amount = 1.0 * cash_balance
			foreign_currency_balance += buy_amount / current_price
			cash_balance -= buy_amount

		elif ( prediction == 0 ):
			sell_amount = 1.0 * foreign_currency_balance
			cash_balance += sell_amount * current_price
			foreign_currency_balance -= sell_amount

	final_balance = cash_balance + ( foreign_currency_balance * current_price )

	if ( view_results ):
		print( 'future prices:' )
		for future_price in future_prices:
			print( future_price )

		print( f'\n\nfinal balance: \n{ final_balance }' )

	return final_balance



#==========================================================
#	MAIN
#==========================================================

if ( __name__ == '__main__' ):
	bin_model = load_model( './models/bin_model.h5' )
	print(f'df:\n{ df }')

	sim_final_balance = simulator( bin_model, df, initial_balance, look_ahead, view_results=True )

