import tensorflow as tf
from tensorflow.keras.losses import Loss

class LatentLoss(Loss):
    def __init__(self, beta=1.0):
        super ().__init__()
        self.beta = beta

    def call(self, dummy_ground_truth, outputs):
        del dummy_ground_truth
        z_e, z_q = tf.split(outputs, 2, axis=-1)
        vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q)**2)
        commit_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q))**2)
        return vq_loss + self.beta * commit_loss