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
from keras.layers import Input, LSTM, Dropout, Dense



#========================================================
#   FUNCTIONS
#========================================================

def show_data_plot( data, label : str, display_data=True ):

    if ( display_data ): print(label, '=', data)

    plt.plot( data, label=label )
    plt.legend()
    plt.grid( True )
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



def create_sequences( data, target ):
    data_seq = []
    target_seq = []
    for i in range( len( data ) ):
        data_seq.append( data[ :i+1 ]  )
        target_seq.append( target[ i ]  )

    return np.array( data_seq ), np.array( target_seq )



#========================================================
#   MAIN
#========================================================

look_ahead = 5
epochs = 100
display_training = True

df = yf.download( 'EURPLN=X', start='2008-01-01', end=None )
df = add_all_ta_features( df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True )
selected_columns = [ 'Close', 'volume_adi', 'momentum_rsi', 'trend_sma_slow' ]
df = df[ selected_columns ]

df[ 'FuturePrice' ] = df['Close'].shift( -look_ahead )
df[ 'PriceChange' ] = df[ 'FuturePrice' ] - df[ 'Close' ]
df[ 'Movement' ] = ( df['PriceChange'] > 0 ).astype( int )

close_data = df[ ['Close'] ]


features = df[ selected_columns ]
target = df[ 'Movement' ]

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform( train_features )

train_features_scaled = scaler.transform( train_features )
test_features_scaled = scaler.transform( test_features )

train_features_reshaped = train_features_scaled.reshape(train_features_scaled.shape[0], train_features_scaled.shape[1], 1)
test_features_reshaped = test_features_scaled.reshape(test_features_scaled.shape[0], test_features_scaled.shape[1], 1)




deep_stock = Sequential([
    Input( shape=( train_features.shape[1], train_features.shape[2] ) ),
    LSTM( 64, return_sequences=True ),
    LSTM( 64 ),
    Dropout( 0.2 ),
    Dense( 32, activation='relu' ),
    Dense( 1, activation='sigmoid' )
])

deep_stock.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = deep_stock.fit( train_features, train_target ,epochs=epochs, batch_size=32, validation_data=( test_features, test_target ) )

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']


if ( display_training ):
    epochs = range( 1, len( loss ) + 1 )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


if ( '__main__' == __name__ ):
    predictions = my_model.predict(test_features_reshaped)

    transformed_predictions = np.repeat(predictions, 4, axis=1)
    transformed_predictions = scaler.transform(transformed_predictions)

    reshaped_predictions = transformed_predictions.reshape(-1, 4)
    predictions_inverse = scaler.inverse_transform(reshaped_predictions)

    print( f'inverse transformation predicitons:\n{ predictions_inverse }' )
    print( f'inverse transformation values:\n{ test_features_scaled }' )

    plt.figure(figsize=(10, 6))

    plt.plot(test_target.index, test_target, label='Actual Values', color='blue')
    plt.plot(test_target.index, predictions_inverse, label='Forecasted Values', color='red')
    plt.title('Actual vs Forecasted Values')
    plt.xlabel('Time/Sequence')
    plt.ylabel('Value')
    plt.legend()
    plt.show()