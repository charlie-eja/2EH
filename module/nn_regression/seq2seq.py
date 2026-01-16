from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np

def seq2seq_model(x : np.ndarray,
                  y : np.ndarray,
                  model_type : str ,
                  x_hidden : int = 40,
                  y_hidden : int = 40,
                  batch_size : int =64,
                  epochs : int =100,
                  verbose : int =1,
                  model_name:str ='model',):
    '''
    :param model_type:
    1. zero
    2. rand_gaussian(Gaussian)
    3. yt
    4. xt_yt
    5. xt_head_yt
    6. yt_head
    7. xt_yt_head
    8. xt_head_yt_head
    '''
    if model_type == 'zero' :
        x,decoder_input,decoder_target=zero(x,y)
    elif model_type == 'rand_gaussian':
        x,decoder_input,decoder_target,x_val,decoder_input_val,decoder_target_val=rand_gaussian(x,y,x_val,y_val)
    elif model_type == 'yt':
        x,decoder_input,decoder_target,x_val,decoder_input_val,decoder_target_val=yt(x,y,x_val,y_val)
    elif model_type == 'xt_yt':
        x,decoder_input,decoder_target,x_val,decoder_input_val,decoder_target_val=xt_yt(x,y,x_val,y_val)
    else :
         return print('model_type not found')

    x_step  = x.shape[1]
    y_step  = decoder_input.shape[1]
    x_dim = x.shape[2]
    y_dim = y.shape[2]
    decoder_input_dim = decoder_input.shape[2]
    x_hidden = x_hidden
    y_hidden = y_hidden

    enc_inputs = layers.Input(shape=(x_step, x_dim), name="encoder_inputs")
    enc_lstm = layers.LSTM(x_hidden, return_state=True, name="encoder_lstm")
    enc_out, enc_h, enc_c = enc_lstm(enc_inputs)
    enc_states = [enc_h, enc_c]

    dec_inputs = layers.Input(shape=(y_step, decoder_input_dim), name="decoder_inputs")
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
    model.save(f"{model_name}.keras")
    return model,history


def zero(x,y):
    x = x
    decoder_input = np.zeros((y.shape[0],y.shape[1],y.shape[2]))
    decoder_target = y
    return  x,decoder_input,decoder_target

def rand_gaussian(x,y):
    x = x
    decoder_input = np.random.randn(y.shape[0],y.shape[1],y.shape[2])
    decoder_target = y
    return  x,decoder_input,decoder_target

def yt(x,y):

    x = x[:,:-1,:]
    input_shape = x.shape[2]-y.shape[2]
    decoder_input = x[:,-1,input_shape:]
    decoder_input = np.repeat(decoder_input[:, None, :], y.shape[1], axis=1)
    decoder_target = y

    return  x,decoder_input,decoder_target

def xt_yt(x,y):
    x = x[:,:-1,:]
    decoder_input = x[:, -1, :]
    decoder_input = np.repeat(decoder_input[:, None, :], y.shape[1], axis=1)
    decoder_target = y
    return  x,decoder_input,decoder_target


def build_inference_models(model, y_dim, hidden):
    # ===== Encoder: X -> (h, c) =====
    encoder_inputs = model.input[0]  # [encoder_inputs, decoder_inputs]
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


def predict_next(x,y,x_hidden,y_hidden,model,model_type,time_step=3):
    x_dim = x.shape[2]
    y_dim = y.shape[2]
    if model_type == 'zero' :
        x = x
        dim = y_dim
        y_now = np.zeros((x.shape[0], 1, dim))
    elif model_type == 'rand_gaussian' :
        x = x
        dim = y_dim
        y_now = np.random.randn(x.shape[0], 1, dim)
    elif model_type == 'yt' :
        x = x[:,:-1,:]
        input_shape = x.shape[2]-y.shape[2]
        y_now = x[:,-1,input_shape:]
        dim = y_dim
    elif model_type == 'xt_yt' :
        x = x[:, :-1, :]
        y_now = x[:, -1, :]
        dim = x_dim

    encoder_model, decoder_model = build_inference_models(model, dim, y_hidden)
    h, c = encoder_model.predict(x)
    y_prev = y_now.reshape(x.shape[0], 1, dim)
    outputs = []
    for _ in range(time_step):
        y_step, h, c = decoder_model.predict([y_prev, h, c])
        outputs.append(y_step)
        if model_type == 'zero':
            y_prev = y_now.reshape(x.shape[0], 1, dim)
        elif model_type == 'rand_gaussian':
            y_now = np.random.randn(x.shape[0], 1, dim)
            y_prev = y_now.reshape(x.shape[0], 1, dim)
        elif model_type == 'yt':
            y_prev = y_now.reshape(x.shape[0], 1, dim)
        elif model_type == 'xt_yt':
            y_prev = y_now.reshape(x.shape[0], 1, dim)

    y_pred = np.concatenate(outputs, axis=1)  # -> (N, time_step, y_dim)
    return y_pred