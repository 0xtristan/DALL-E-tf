import attr
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Layer, Lambda, MaxPool2D
from tensorflow.keras.activations import swish
from tensorflow.keras import Model, Sequential

from collections import OrderedDict
from functools import partial

# @attr.s(eq=False, repr=False)
# class Encoder(Model):
#     group_count: int = 4
#     n_hid: int = attr.ib(default=256, validator=lambda i, a, x: x >= 64)
#     n_blk_per_group: int = attr.ib(default=2, validator=lambda i, a, x: x >= 1)
#     input_channels: int = attr.ib(default=3, validator=lambda i, a, x: x >= 1)
#     vocab_size: int = attr.ib(default=8192, validator=lambda i, a, x: x >= 512)
#
#     def __attrs_post_init__(self):
#         super().__init__()
#         self.blk_range = range(self.n_blk_per_group)
#         self.n_layers = self.group_count * self.n_blk_per_group
#
#         self.input_conv = Conv2D(self.n_hid, 7, input_shape=(256, 256, self.input_channels))
#         # self.block = EncoderBlock(1 * self.n_hid, self. n_layers)
#
#     def call(self, x):
#         x = self.input_conv(x)
#         # x = self.block(x)
#         # x = Sequential([EncoderBlock(1 * self.n_hid, self.n_layers) for i in self.blk_range])(x)
#         return x

# @attr.s(eq=False, repr=False)
# class EncoderBlock(Layer):
#     n_out: int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 == 0)
#     n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)
#     activation: str = attr.ib(default='relu')
#
#     def __attrs_post_init__(self):
#         super().__init__()
#         self.n_hid = self.n_out // 4
#         self.post_gain = 1 / (self.n_layers ** 2)
#         self.activation = tf.keras.activations.get(self.activation)
#
#     def call(self, x):
#         x_res = Conv2D(self.n_out, 1)(x) if self.n_out != x.shape[-1] else tf.identity(x)
#         for i in range(2):
#             x = Conv2D(self.n_hid, 3, padding='same', activation=self.activation)(x)
#         x = Conv2D(self.n_out, 1)(x)
#         return x_res + self.post_gain * x

# @attr.s(eq=False, repr=False)
# class EncoderBlock(Model):
#     n_out: int = attr.ib(validator=lambda i, a, x: x >= 1 and x % 4 == 0)
#     n_layers: int = attr.ib(validator=lambda i, a, x: x >= 1)
#     activation: str = attr.ib(default='relu')
#
#     def __attrs_post_init__(self):
#         super().__init__()
#         self.n_hid = self.n_out // 4
#         self.post_gain = 1 / (self.n_layers ** 2)
#         self.activation = tf.keras.activations.get(self.activation)
#
#         self.res_conv = Conv2D(self.n_out, 1)
#         self.conv = Conv2D(self.n_hid, 3, padding='same', activation=self.activation)
#         self.out_conv = Conv2D(self.n_out, 1)
#
#     def call(self, x):
#         x_res = self.res_conv(x) if self.n_out != x.shape[-1] else tf.identity(x)
#         for i in range(2):
#             x = self.conv(x)
#         x = self.out_conv(x)
#         return x_res + self.post_gain * x

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

def encoder(img_size=None, group_count=4, n_hid=256, n_blk_per_group=2, vocab_size=8192, activation='relu'):
    act_fn = tf.keras.activations.get(activation)
    x_in = Input(shape=(img_size, img_size, 3))
    x = Conv2D(n_hid, 7, padding='same')(x_in)
    for i in range(group_count):
        for _ in range(n_blk_per_group):
            x = encoder_block(x, 256 * 2**i, n_hid, activation=activation)
        if i < group_count - 1:
            x = MaxPool2D(2)(x)
    x = act_fn(x)
    x = Conv2D(vocab_size, 1, dtype=tf.float32)(x)
    return Model(x_in, x)


eb = encoder(256)
eb.summary(line_length=200)