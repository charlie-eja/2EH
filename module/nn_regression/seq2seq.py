from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def seq2seq_model(x : np.ndarray,
                  y : np.ndarray,
                  x_hidden : int = 40,
                  y_hidden : int = 40,
                  batch_size : int =64,
                  epochs : int =100,
                  verbose : int =1,
                  model_type : str ='zero',):
    '''
    :param model_type:
    1. zero
    2. randn(Gaussian)
    3. yt
    4. xt_yt
    5. xt_head_yt
    6. yt_head
    7. xt_yt_head
    8. xt_head_yt_head
    '''
    if model_type == 'zero' :
        x=x
    elif model_type == 'randn':
        x=x
    decoder_input  = y[:, :-1, :]
    decoder_target = y[:, 1:, :]
    # decoder_target = y[:, :-1, :]
    x_step  = x.shape[1]
    y_step  = decoder_input.shape[1]
    x_dim = x.shape[2]
    y_dim = y.shape[2]
    x_hidden = x_hidden
    y_hidden = y_hidden

    enc_inputs = layers.Input(shape=(x_step, x_dim), name="encoder_inputs")
    enc_lstm = layers.LSTM(x_hidden, return_state=True, name="encoder_lstm")
    enc_out, enc_h, enc_c = enc_lstm(enc_inputs)
    enc_states = [enc_h, enc_c]

    dec_inputs = layers.Input(shape=(y_step, y_dim), name="decoder_inputs")
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

    history=model.fit([x, decoder_input],
              decoder_target,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              callbacks=[early_stop],)
    return model,history


def build_inference_models(model, y_dim, hidden):
    # ===== Encoder: X -> (h, c) =====
    encoder_inputs = model.input[0]  # [encoder_inputs, decoder_inputs] 的第 0 個
    _, state_h, state_c = model.get_layer("encoder_lstm").output
    encoder_model = Model(encoder_inputs, [state_h, state_c])

    # ===== Decoder step: (y_prev, h, c) -> (y_step, h_new, c_new) =====
    dec_input_y = layers.Input(shape=(1, y_dim), name="dec_step_input_y")
    dec_input_h = layers.Input(shape=(hidden,), name="dec_step_input_h")
    dec_input_c = layers.Input(shape=(hidden,), name="dec_step_input_c")

    dec_lstm   = model.get_layer("decoder_lstm")
    dec_dense  = model.get_layer("decoder_output")

    dec_out, dec_h, dec_c = dec_lstm(
        dec_input_y,
        initial_state=[dec_input_h, dec_input_c]
    )
    y_step = dec_dense(dec_out)  # shape = (batch, 1, y_dim)

    decoder_model = Model(
        [dec_input_y, dec_input_h, dec_input_c],
        [y_step, dec_h, dec_c]
    )

    return encoder_model, decoder_model

def predict_next(x_seq, y_now, encoder_model, decoder_model, y_dim,time_step=3):

    h, c = encoder_model.predict(x_seq)
    batch = x_seq.shape[0]
    y_prev = y_now.reshape(batch, 1, y_dim)

    outputs = []
    for _ in range(time_step):
        y_step, h, c = decoder_model.predict([y_prev, h, c])
        outputs.append(y_step)
        y_prev = y_step

    y_pred = np.concatenate(outputs, axis=1)  # -> (N, time_step, y_dim)
    return y_pred
