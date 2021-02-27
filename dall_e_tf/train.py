import tensorflow as tf
from tensorflow.keras import Model

from dall_e_tf.encoder import dvae_encoder
from dall_e_tf.decoder import dvae_decoder
from dall_e_tf.utils import plot_reconstructions

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

train_size = 60000
batch_size = 32
test_size = 10000

vocab_size = 8192
n_hid = 128
enc = dvae_encoder(group_count=2, n_hid=n_hid, n_blk_per_group=2, input_channels=x_train.shape[-1], vocab_size=vocab_size)
dec = dvae_decoder(group_count=2, n_init=n_hid//2, n_hid=n_hid, n_blk_per_group=2, output_channels=x_train.shape[-1], vocab_size=vocab_size)

vae = Model(enc.input, dec(enc.output))
vae.summary(line_length=200)

vae.compile(loss='mse', optimizer='adam')

vae.fit(x_train, x_train,
        validation_data=(x_test, x_test),
        steps_per_epoch=10,
        epochs=1,
        # batch_size=batch_size,
        shuffle=True)

plot_reconstructions(vae, x_test)