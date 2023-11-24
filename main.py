#========================================================
#   IMPORTS
#========================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yfinance as yf

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
epochs = 10

display_training = False
display_results = False

scaler = MinMaxScaler()

df = yf.download( 'EURPLN=X', start='2008-01-01', end=None )
df = df[ [ 'Close' ] ]


df[ 'FutureClose' ] = df[ 'Close' ].shift( -look_ahead )
df[ 'Direction' ] = np.where( df['FutureClose'] > df['Close'], 1, 0 )
df = df.dropna()

scaled_data = scaler.fit_transform( df )
X_class = scaled_data[ :, 0 ]       # close price
y_class = scaled_data[ :, 2 ]       # direction (binary target)

X_class = X_class.reshape( -1, 1, 1 )

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split( X_class, y_class, test_size=0.2, random_state=42 )


print( f'scaled data:\n{ scaled_data }' )


binary_model = Sequential([
    Input( shape=( X_train_class.shape[-1], X_train_class.shape[2] ) ),
    LSTM( 64, return_sequences=True ),
    LSTM( 64 ),
    Dense( 64, activation='relu' ),
    Dense( 1, activation='sigmoid' )
], name='binary_model')


binary_model.compile( optimizer='adam', loss='binary_crossentropy', metrics=[ 'accuracy' ] )
binary_model.fit(X_train_class, y_train_class, epochs=10, batch_size=32, validation_data=(X_test_class, y_test_class))


direction_preds = binary_model.predict( X_class )
direction_preds = np.round( direction_preds )
df['PredictedDirection'] = direction_preds

print( f'direction_preds:\n{ direction_preds }' )
print( f'Dataframe:\n{ df }' )


plt.plot( df.index, df[ 'Direction' ], label='actual direction', marker='o' )
plt.plot( df.index, direction_preds, label='Predicted direction', marker='x', linestyle='--' )
plt.title('Binary Classification: Actual vs Predicted Directions')
plt.xlabel('Time/Sequence')
plt.ylabel('Direction (0: Down, 1: Up)')
plt.legend()
plt.show()



X_reg = df[ ['Close', 'PredictedDirection'] ].values
y_reg = scaled_data[ :, 1 ]     # target: FutureClose

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

print( f'X_reg\n{ X_reg }' )
print( f'y_reg\n{ y_reg }' )


binary_model = Sequential([
    Input( shape=( X_train_class.shape[-1], X_train_class.shape[2] ) ),
    LSTM( 128, return_sequences=True ),
    LSTM( 64 ),
    Dense( 64, activation='relu' ),
    Dense( 1, activation='sigmoid' )
], name='binary_model')

binary_model.compile( optimizer='adam', loss='binary_crossentropy', metrics=[ 'accuracy' ] )
binary_model.fit(X_train_class, y_train_class, epochs=150, batch_size=32, validation_data=(X_test_class, y_test_class))


direction_preds = binary_model.predict( X_class )
direction_preds = np.round( direction_preds )
df['PredictedDirection'] = direction_preds

plt.plot( df.index[ -100: ], direction_preds[ -100: ], label='Predicted direction', marker='x', linestyle='--' )
plt.plot( df.index[ -100: ], df[ 'Direction' ].tail( 100 ), label='actual direction', marker='o' )
plt.title('Binary Classification: Actual vs Predicted Directions')
plt.xlabel('Time/Sequence')
plt.ylabel('Direction (0: Down, 1: Up)')
plt.legend()
plt.show()

print( f'direction_preds:\n{ direction_preds }' )
print( f'Dataframe:\n{ df }' )

