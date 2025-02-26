#========================================================
#   	IMPORTS
#========================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras import Sequential
from keras.layers import Input, LSTM, Dropout, Dense
from keras.models import load_model



#=======================================================
#	GLOBALS
#=======================================================

look_back = 90
look_ahead = 10

epochs = 1000
scaler = MinMaxScaler()

df = yf.download( "EURPLN=X", end="2090-01-01" )
start_date = datetime( 2021, 1, 1 )

display_training = False
display_results = False

test_quarter_gains = {}



#========================================================
#   	FUNCTIONS
#========================================================

def show_data_plot( data, label : str, display_data=True ):

    if ( display_data ): print(label, '=', data)

    plt.plot( data, label=label )
    plt.legend()
    plt.grid( True )
    plt.show()

    return None



def show_two_data_plots( data_01, data_02, label_01 : str, label_02 : str, display_data=True ):

    fig, axs = plt.subplots( 2 )

    if ( display_data ):
        print( label_01, "=", data_01 )
        print( label_02, "=", data_02 )

    axs[0].plot( data_01, label=label_01 )
    axs[0].legend()
    axs[0].grid( True )

    axs[1].plot( data_02, label=label_02 )
    axs[1].legend()
    axs[1].grid( True )

    plt.show()

    return None



def display_prediction_stats( df, scaler, model ):


	close_values = df[[ 'Close' ]].values.reshape( -1, 1 )
	close_values_scaled = scaler.transform( close_values )
	print( close_values )

	predictions = model.predict( close_values_scaled )
	predictions = scaler.inverse_transform( predictions )

	df[ 'PredictedClose' ] = predictions

	plt.plot( close_values )
	plt.savefig( "./img/predictions.png" )

	return df



def create_sequences( data, target ):

    data_seq = []
    target_seq = []
    for i in range( len( data ) ):
        data_seq.append( data[ :i+1 ]  )
        target_seq.append( target[ i ]  )

    return np.array( data_seq ), np.array( target_seq )



def display_dataset( dataset ):

	print( f'dataset:\n{ dataset }' )

	return



def display_diagnostics( epoch_count : int, history, save_path : str ):
    for epoch in range(0, epoch_count, epoch_count ):
        plt.figure( figsize=(10, 5) )

        plt.subplot( 1, 2, 1 )
        plt.plot( history.history[ 'accuracy' ], label='Training Accuracy')
        plt.plot( history.history[ 'val_accuracy' ], label='Validation Accuracy')
        plt.title( 'Training and Validation Accuracy' )
        plt.xlabel( 'Epochs' )
        plt.ylabel( 'Accuracy' )
        plt.grid()
        plt.legend()

        plt.subplot( 1, 2, 2 )
        plt.plot(history.history[ 'loss' ], label='Training Loss')
        plt.plot(history.history[ 'val_loss' ], label='Validation Loss')
        plt.title( 'Training and Validation Loss' )
        plt.xlabel( 'Epochs' )
        plt.ylabel( 'Loss' )
        plt.legend()
        plt.grid()
        plt.savefig( save_path )

        plt.tight_layout()
        plt.show()

    return



def get_xy_classes( df, look_back : int, view_data=True ):

	"""
	:param df:              dataframe,
	:look_back:             number of data considered when making a prediction,
	:param look_ahead:      number of data points to look into the future;

	returns:                X_class, y_class;

	"""

	df = df[ [ 'Close' ] ]
	X_class = []
	y_class = []

	df[ 'Direction' ] = ( df[ 'Close' ].shift(-1) > df[ 'Close' ] ).astype( int )
	df.dropna( inplace=True )   # last column

	for i in range( len( df ) - look_back - 1 ):
		X_class.append( df[ 'Close' ][ i : ( i+look_back )].values )
		y_class.append( df[ 'Direction' ][ i + look_back ] )


	X_class = np.array( X_class )
	y_class = np.array( y_class )
	y_class = y_class.reshape( -1, 1 )

	if ( len( X_class ) == len( y_class ) ): print( "lengths match" )
	else: print( "lengths dont match" )

	if ( view_data ):
			print( f'dataframe:\n{ df }' )
			print( f'X_class:\n{ X_class }\n length:{ len(X_class) }\ny_class:\n{ y_class }\nlength: { len( y_class ) }' )

	print( f'X_class shape: { X_class.shape }\ny_class shape: { y_class.shape }' )

	return  X_class, y_class



def train_model( df, scaler, look_ahead : int, train_model=True ):

	"""
	:param scaler:		for scaling data,
	:param look_ahead:	looking some points into the future;

	returns:		binary_model, history, X_train_class, X_test_class, y_train_class, y_test_class;
	"""

	X_class, y_class = get_xy_classes( df, scaler, look_ahead, view_data=True )

	X_class_scaled = scaler.fit_transform( X_class )            # X_class
	X_class_reshaped = X_class_scaled.reshape( X_class_scaled.shape[ 0 ] , X_class_scaled.shape[ 1 ] , 1 )
	print( f'X_class_scaled shape: { X_class_scaled.shape } ' )
	print( f'X_class_reshaped shape: { X_class_reshaped.shape }' )
    
    
	y_class_scaled = scaler.fit_transform( y_class )            # y_class
	y_class_reshaped = y_class_scaled.reshape( -1, 1, 1 )
	print( f'y_class_scaled shape: { y_class_scaled.shape } ' )
	print( f'y_class_reshaped shape: { y_class_reshaped.shape }' )


	X_class_scaled = scaler.fit_transform( X_class )
	X_class_reshaped = X_class_scaled.reshape( -1, 1, 1 )

	X_train_class, X_test_class, y_train_class, y_test_class = train_test_split( X_class_reshaped, y_class, test_size=0.2, random_state=42 )

	binary_model = Sequential([
		Input( shape=( X_train_class.shape[ -1 ], X_train_class.shape[ 2 ] ) ),
        	LSTM( 128, return_sequences=True ),
		LSTM( 64 ),
        	Dense( 64, activation='relu' ),
        	Dense( look_ahead, activation='sigmoid' )
	])


	if ( train_model ):
		binary_model.compile( optimizer='adam', loss='binary_crossentropy', metrics=[ 'accuracy' ] )
		history = binary_model.fit( X_train_class, y_train_class, batch_size=64, epochs=epochs, validation_data=( X_test_class, y_test_class ) )
		binary_model.save( './models/bin_model.h5' )
		display_diagnostics( epochs, history, "./img/diagnostics_plot.png" )

		display_dataset( df )

		return binary_model, history, X_train_class, X_test_class, y_train_class, y_test_class

	return binary_model, X_train_class, X_test_class, y_train_class, y_test_class



#========================================================
#   MAIN
#========================================================

if (__name__ == '__main__'):

	binary_model, X_train_class, X_test_class, y_train_class, y_test_class = train_model( df, scaler, look_ahead=look_ahead, train_model=False )

	X_train_class = X_train_class.reshape( -1, X_train_class.shape[ -1 ] )

	scaler.fit( X_train_class )
	ds_model = load_model( './models/bin_model.h5' )
	display_prediction_stats( df, scaler, ds_model )


	'''
	print( f"x_test_class:\n{ X_test_class }" )
	print( f"X_train_class:\n{ X_train_class }" )

	y_out = binary_model.predict( X_test_class )
	y_out = scaler.inverse_transform( y_out )
	y_pred_bin = ( y_out > 0.5 ).astype( int )

	print( f'y_out:\n{ y_out }' )
	print( f'preds_bin:\n{ y_pred_bin }' )


	plt.plot( y_pred_bin[ -50: ], label='predicted (binary)', color='blue', linestyle='--' )
	plt.plot( y_test_class[ -50: ], label='actual values (binary)', color='red' )
	plt.legend()
	plt.savefig( './img/testing_plot.png' )
	plt.show()
	'''
