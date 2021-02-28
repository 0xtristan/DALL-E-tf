import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D
from tensorflow.keras import Model

def decoder_block(x_in, n_out, n_layers, activation='relu'):
    n_hid = n_out // 4
    post_gain = 1 / (n_layers ** 2)
    act_fn = tf.keras.activations.get(activation)
    id_path = Conv2D(n_out, 1) if n_out != x_in.shape[-1] else tf.identity

    x = act_fn(x_in)
    x = Conv2D(n_hid, 1)(x)
    for i in range(2):
        x = act_fn(x)
        x = Conv2D(n_hid, 3, padding='same')(x)
    x = act_fn(x)
    x = Conv2D(n_out, 3, padding='same')(x)
    return id_path(x_in) + post_gain * x

def dvae_decoder(img_size=None, group_count=4, n_init=128, n_hid=256, n_blk_per_group=2, output_channels=3,
                 vocab_size=8192, activation='relu'):
    act_fn = tf.keras.activations.get(activation)
    n_layers = group_count * n_blk_per_group
    x_in = Input(shape=(img_size, img_size, vocab_size))
    x = Conv2D(n_init, 1, dtype=tf.float32)(x_in)
    for i in reversed(range(group_count)):
        for _ in range(n_blk_per_group):
            x = decoder_block(x, n_hid * 2**i, n_layers, activation=activation)
        if i > 0:
            x = UpSampling2D(2)(x)
    x = act_fn(x)
    x = Conv2D(output_channels, 1, dtype=tf.float32)(x) # Todo: figure out why this is 2x output channels
    return Model(x_in, x)