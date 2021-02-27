import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D
from tensorflow.keras import Model

def encoder_block(x_in, n_out, n_layers, activation='relu'):
    n_hid = n_out // 4
    post_gain = 1 / (n_layers ** 2)
    act_fn = tf.keras.activations.get(activation)
    id_path = Conv2D(n_out, 1) if n_out != x_in.shape[-1] else tf.identity

    x = act_fn(x_in)
    for i in range(3):
        x = Conv2D(n_hid, 3, padding='same', activation=activation)(x)
    x = Conv2D(n_out, 1, activation=activation)(x)
    return id_path(x_in) + post_gain * x

def dvae_encoder(img_size=None, group_count=4, n_hid=256, n_blk_per_group=2, input_channels=3, vocab_size=8192,
                 activation='relu'):
    act_fn = tf.keras.activations.get(activation)
    n_layers = group_count * n_blk_per_group
    x_in = Input(shape=(img_size, img_size, input_channels))
    x = Conv2D(n_hid, 7, padding='same')(x_in)
    for i in range(group_count):
        for _ in range(n_blk_per_group):
            x = encoder_block(x, n_hid * 2**i, n_layers, activation=activation)
        if i < group_count - 1:
            x = MaxPool2D(2)(x)
    x = act_fn(x)
    x = Conv2D(vocab_size, 1, dtype=tf.float32)(x)
    return Model(x_in, x)