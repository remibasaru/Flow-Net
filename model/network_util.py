import tensorflow as tf

_BATCH_NORM_EPSILON = 5e-5
_BATCH_NORM_DECAY = 0.997


def relu6(x, name='relu6'):
	return tf.nn.relu6(x, name)


def LeakyReLU(x, leak=0.1, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1.0 + leak)
		f2 = 0.5 * (1.0 - leak)
		return f1 * x + f2 * abs(x)


def batch_norm(inputs, training, axis=-1, scale=False, name=None):
	# We set fused=True for a significant performance boost. See
	# https://www.tensorflow.org/performance/performance_guide  # common_fused_ops

	return tf.layers.batch_normalization(inputs=inputs, axis=axis, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
										 center=True, scale=scale, training=training, fused=True, name=name)


def deconv2d(input_, output_dim, filter_size, stride, name='convT2d'):

	with tf.variable_scope(name):

		deconv = tf.layers.conv2d_transpose(input_, output_dim, [filter_size, filter_size],
											strides=stride, padding='SAME')

		return deconv


def deconv2d_block(input_, output_dim, filter_size, stride, name='convT2d'):

	with tf.variable_scope(name):

		deconv = deconv2d(input_, output_dim, filter_size, stride, name)
		deconv = tf.nn.relu(deconv)
	return deconv


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=True, _last_dim=None):
	with tf.variable_scope(name):
		if _last_dim is None:
			_last_dim = input_.get_shape()[-1]
		w = tf.get_variable('w', [k_h, k_w, _last_dim, output_dim],
							regularizer=tf.contrib.layers.l2_regularizer(_BATCH_NORM_DECAY),
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
		if bias:
			biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.bias_add(conv, biases)

		return conv


def conv2d_block(input, out_dim, k, s, is_train, name, last_dim=None):
	with tf.name_scope(name), tf.variable_scope(name):
		net = conv2d(input, out_dim, k, k, s, s, name='conv2d', _last_dim=last_dim)
		# net = batch_norm(net, training=is_train)
		# net = relu6(net)
		net = tf.nn.relu(net)
		return net


def conv_1x1(input, output_dim, name, bias=False):
	with tf.name_scope(name):
		return conv2d(input, output_dim, 1, 1, 1, 1, stddev=0.02, name=name, bias=bias)


def pwise_block(input, output_dim, is_train, name, bias=False, scale=False, bnorm=True, relu=True):
	with tf.name_scope(name), tf.variable_scope(name):
		out = conv_1x1(input, output_dim, bias=bias, name='pwb')
		if bnorm:
			out = batch_norm(out, training=is_train, scale=scale)
		if relu:
			out = relu6(out)
		return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier=1, strides=None, padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
	if strides is None:
		strides = [1, 1, 1, 1]
	with tf.variable_scope(name):
		in_channel=input.get_shape().as_list()[-1]
		w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
							regularizer=tf.contrib.layers.l2_regularizer(_BATCH_NORM_DECAY),
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None, name=None, data_format=None)
		if bias:
			biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
			conv = tf.nn.bias_add(conv, biases)

		return conv


def average_endpoint_error(labels, predictions):
	"""
	Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
		sqrt[sum_across_channels{(X - Y)^2}]
	"""
	num_samples = predictions.shape.as_list()[0]
	with tf.name_scope(None, "average_endpoint_error", (predictions, labels)) as scope:
		predictions = tf.to_float(predictions)
		labels = tf.to_float(labels)
		predictions.get_shape().assert_is_compatible_with(labels.get_shape())

		squared_difference = tf.square(tf.subtract(predictions, labels))
		# sum across channels: sum[(X - Y)^2] -> N, H, W, 1
		loss = tf.reduce_sum(squared_difference, 3, keepdims=True)
		loss = tf.sqrt(loss)
		return tf.reduce_sum(loss) / num_samples, squared_difference