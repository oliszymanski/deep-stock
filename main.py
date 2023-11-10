#========================================================
#   IMPORTS
#========================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yfinance as yf

from ta import add_all_ta_features

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras import Sequential
from keras.layers import LSTM, Dropout, LeakyReLU, Dense



#========================================================
#   FUNCTIONS
#========================================================

def show_data_plot( data, label : str, display_data=True ):

    if ( display_data ): print(label, '=', data)

    plt.plot( data, label=label )
    plt.legend()
    plt.grid(True)
    plt.show()

    return None



def show_two_data_plots( data_01, data_02, label_01 : str, label_02 : str, display_data=True ):

    fig, axs = plt.subplots(2)

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



def create_sequences( data, lookback ):
    input_sequences = []
    target_sequences = []

    for i in range( len( data ) - lookback ):
        input_sequences.append(  data[ i : i + lookback ]  )
        target_sequences.append(  data[ i + lookback ]  )

    return np.array( input_sequences ), np.array( target_sequences )



#========================================================
#   MAIN
#========================================================

lookback = 30

df = yf.download( 'EURPLN=X', start='2008-01-01', end=None )

df = add_all_ta_features( df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True )

selected_columns = ['Close', 'volume_adi', 'momentum_rsi', 'trend_sma_slow']
df = df[selected_columns]



close_data = df[ ['Close'] ]


train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)

# show_data_plot( train_data, 'data for training', display_data=True )

num_time_steps = train_data.shape[0]
num_features = train_data.shape[1]

scaler = MinMaxScaler()
scaler.fit(train_data)
train_data_norm = scaler.transform(train_data)    # Normalize training and testing data
test_data_norm = scaler.transform(test_data)

train_input, train_target = create_sequences( train_data_norm, lookback )    # creating inputs and targets
test_input, test_target = create_sequences( test_data_norm, lookback )


print( "train_data_norm shape =", train_data_norm.shape )
print( "test_data_norm shape =", test_data_norm.shape )




deep_stock = Sequential( [
    LSTM(64, return_sequences=True, input_shape=(train_input.shape[1], train_input.shape[2])),
    LeakyReLU( alpha=0.01 ),
    LSTM( 128, return_sequences=True ),
    LSTM( 128, return_sequences=True ),
    LeakyReLU( alpha=0.01 ),
    LSTM( 128, return_sequences=True ),
    LSTM( 128, return_sequences=True ),
    LeakyReLU( alpha=0.01 ),
    LSTM( 128, return_sequences=True ),
    LeakyReLU( alpha=0.01),
    LSTM( 128, return_sequences=True ),
    LeakyReLU( alpha=0.01 ),
    Dropout( 0.2 ),
    LSTM( 128, return_sequences=False ),
    LeakyReLU( alpha=0.01 ),
    Dense( 1 )
], name="deep_stock" )


deep_stock.compile( optimizer='adam', metrics=['accuracy'], loss='mean_squared_error' )
deep_stock.fit( train_input, train_target, epochs=12, batch_size=32 )
loss = deep_stock.evaluate( test_input, test_target )

predictions = deep_stock.predict( test_input )
predictions = predictions.reshape( -1, 1 )
predictions = scaler.inverse_transform( predictions )

# average_predictions = predictions.mean( axis=1 )
# average_predictions = scaler.inverse_transform( average_predictions )

# average_predictions = scaler.inverse_transform( average_predictions )
# print( 'average_predictions =', average_predictions )

# print('predictions shape =', average_predictions.shape)
# print("predictions =", average_predictions)

train_data = np.array( scaler.inverse_transform( test_target ) )

plt.plot( train_data, label='actual data' )
plt.title( 'price of EUR/PLN as from 2020-01-01' )
plt.xlabel( 'Time point' )

show_data_plot( predictions, 'predictions', display_data=True )


if ( '__main__' == __name__ ):
    print( f'test input \n{ test_input }' )
    print( f'predictions:\n{ predictions }' )