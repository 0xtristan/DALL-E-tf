import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

logit_laplace_eps: float = 0.1

def map_pixels(x: tf.Tensor) -> tf.Tensor:
	if len(x.shape) != 4:
		raise ValueError('expected input to be 4d')
	if x.dtype != tf.float:
		raise ValueError('expected input to have type float')

	return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps

def unmap_pixels(x: tf.Tensor) -> tf.Tensor:
	if len(x.shape) != 4:
		raise ValueError('expected input to be 4d')
	if x.dtype != tf.float:
		raise ValueError('expected input to have type float')

	return tf.clip_by_value((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)

def plot_reconstructions(model, x_test, n=10):
	decoded_imgs = model(x_test).numpy()
	plt.figure(figsize=(20, 4))
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(np.squeeze(x_test[i]))
		plt.title("original")
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(np.squeeze(decoded_imgs[i]))
		plt.title("reconstructed")
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()