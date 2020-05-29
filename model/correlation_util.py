import os
import numpy as np
import tensorflow as tf


def img_to_patches(raw_input, _patch_size=(128, 128), _stride=100):

	with tf.variable_scope('im2_patches'):
		patches = tf.image.extract_image_patches(
			images=raw_input,
			ksizes=[1, _patch_size[0], _patch_size[1], 1],
			strides=[1, _stride, _stride, 1],
			rates=[1, 1, 1, 1],
			padding='SAME'
		)

		h = tf.shape(patches)[1]
		w = tf.shape(patches)[2]
		patches = tf.reshape(patches, (tf.shape(patches)[0], -1, _patch_size[0],  _patch_size[1], raw_input.shape[-1]))
		patches = tf.transpose(patches, (2, 3, 4, 0, 1))
	return patches, (h, w)


def correlation(input_a, input_b, kernel_size, max_displacement, stride_1, stride_2, padding='VALID'):
	"""
	Function to correlation between pair of feature map as proposed in https://arxiv.org/pdf/1504.06852.pdf.

	:param input_a: stationary input feature map
	:param input_b: convolving input feature map
	:param kernel_size: kernel window size
	:param max_displacement: maximum displacement around centroid patch
	:param stride_1: stride to quantize patches in 'input_a'
	:param stride_2: stride to quantize spatial shift on patches in 'input_b' around patches in 'input_a'
	:param padding:
	:return: resulting feature map from correlation
	"""
	assert kernel_size % 2 == 1, "'kernel_size' must be odd."
	assert max_displacement % stride_2 == 0, "'maximum_displacement' must be divisible by 'stride_2'."
	num_in_channels = input_a.shape[-1]
	K = kernel_size
	k = (K - 1) / 2
	k_s2 = int((max_displacement + k) * 2 + 1)
	with tf.variable_scope('corr_func'):
		input_patch_a, _ = img_to_patches(input_a, _patch_size=(K, K), _stride=stride_1)
		input_patch_a = tf.reshape(input_patch_a, shape=(tf.shape(input_patch_a)[0], tf.shape(input_patch_a)[1], -1, 1))

		input_patch_b, sz = img_to_patches(input_b, _patch_size=(k_s2, k_s2), _stride=stride_1)
		input_patch_b = tf.reshape(input_patch_b, shape=(1, tf.shape(input_patch_b)[0], tf.shape(input_patch_b)[1], -1))

		correlation_output = tf.nn.depthwise_conv2d(input_patch_b, input_patch_a, [1, stride_2, stride_2, 1],
													padding, rate=None, name=None, data_format=None)
		correlation_output = tf.reshape(correlation_output,
										shape=(tf.shape(correlation_output)[1] * tf.shape(correlation_output)[2],
											   num_in_channels, -1, sz[0], sz[1]))
		correlation_output = tf.transpose(correlation_output, (2, 3, 4, 0, 1))
		correlation_output = tf.reduce_sum(correlation_output, axis=-1)

	dim = np.power((2 * max_displacement / stride_2) + 1, 2)
	return correlation_output, int(dim)


if __name__ == "__main__":

	num_in_channel = 3
	bs = 1
	img_a = np.random.randint(10, size=(bs, 225, 125, num_in_channel))
	img_b = np.random.randint(10, size=(bs, 225, 125, num_in_channel))

	# cc = correlation(conv_a_3, conv_b_3, 1, 20, 1, 2)

	img_1_feature_map = tf.placeholder(dtype=tf.float32, shape=(bs, 225, 125, num_in_channel), name="input_img")
	img_2_feature_map = tf.placeholder(dtype=tf.float32, shape=(bs, 225, 125, num_in_channel), name="input_img")

	output = correlation(img_1_feature_map, img_2_feature_map, 3, 20, 1, 2)
	with tf.Session() as sess:
		arg0, arg1, arg2 = sess.run([img_1_feature_map, img_2_feature_map, output], feed_dict={img_1_feature_map: img_a, img_2_feature_map: img_b})

		print(arg0)
