#========================================================
#	IMPORTS
#========================================================

from datetime import datetime, timedelta

import main
from main import train_model
import simulator

import yfinance

import pandas as pd



#========================================================
#	FUNCTIONS
#========================================================

def test_gains( df, start_date, end_date, steps: int ):

	"""
	:param df:		dataframe with 'index', 'Close', 'FutureClose' and 'Direction',
	:param start_date:	starting date point (turned into datetime),
	:param end_date:	end date point (datetime object),
	"""

	ls_final_balances = []

	"""
        while start_date <= end_date:
                train model with the data up to start_date index
                simulate gains on the rest of df
                append gains into ls_final_balances
                start the cycle again
	"""


	step = timedelta( months=3 )


	if ( view_results ):
		print( 'final results' )

	return None



#========================================================
#	MAIN
#========================================================
