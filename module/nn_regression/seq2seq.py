from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def seq2seq_model(X : np.ndarray,
                  Y : np.ndarray,
                  x_hidden : int = 4,
                  y_hidden : int = 4,
                  batch_size : int =64,
                  epochs : int =100,
                  verbose : int =1):

    decoder_input  = Y[:, :-1, :]
    decoder_target = Y[:, 1:, :]
    Tx  = X.shape[1]
    Ty  = decoder_input.shape[1]
    x_dim = X.shape[2]
    y_dim = Y.shape[2]
    x_hidden = x_hidden
    y_hidden = y_hidden

    enc_inputs = layers.Input(shape=(Tx, x_dim), name="encoder_inputs")
    enc_lstm = layers.LSTM(x_hidden, return_state=True, name="encoder_lstm")
    enc_out, enc_h, enc_c = enc_lstm(enc_inputs)
    enc_states = [enc_h, enc_c]

    dec_inputs = layers.Input(shape=(Ty, y_dim), name="decoder_inputs")
    dec_lstm = layers.LSTM(y_hidden, return_sequences=True, return_state=True, name="decoder_lstm")
    dec_out, _, _ = dec_lstm(dec_inputs, initial_state=enc_states)

    dec_dense = layers.TimeDistributed(layers.Dense(y_dim), name="decoder_output")
    dec_outputs = dec_dense(dec_out)

    model = Model([enc_inputs, dec_inputs], dec_outputs)
    model.summary()

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True
    )

    history=model.fit([X, decoder_input],
              decoder_target,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              callbacks=[early_stop],)
    return model,history
