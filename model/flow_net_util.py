import os

import tensorflow as tf

from model.correlation_util import correlation
from model.model_builder import Network
from model.network_util import conv2d_block, conv2d, deconv2d, deconv2d_block, average_endpoint_error, LeakyReLU


class FlowNet(Network):
	def __init__(self, graph_path_train='', graph_path_pred='', device="/cpu:0", classic_mode=True, batch_size=1):
		super().__init__(graph_path_train, graph_path_pred, device)
		self._device = device
		self.output_node_names = []
		self.batch_size = batch_size
		self.classic_mode = classic_mode

		self.input_img_1 = None
		self.input_img_2 = None
		self.correlation_param = {
			"stride_1": 1,
			"stride_2": 2,
			"kernel_size": 1,
			"max_displacement": 20
		}

	def handle_admin(self):
		w = 512
		h = 384
		bs = self.batch_size
		self.input_img_1 = tf.placeholder(dtype=tf.float32, shape=[bs, h, w, 3], name="input_img_1")
		self.input_img_2 = tf.placeholder(dtype=tf.float32, shape=[bs, h, w, 3], name="input_img_2")
		self.target_img = tf.placeholder(dtype=tf.float32, shape=[bs, h, w, 2],
										 name="target_img")

	def build_correlation_stage(self):
		# First Image Channel

		net = conv2d_block(self.input_img_1, 64, 3, 1, self.train_flag, name='conv1_1a')
		self.layers['conv1'] = net  # 1/1 X
		net = conv2d_block(net, 64, 3, 2, self.train_flag, name='conv1_2a')
		with tf.variable_scope('conv2a'):
			net = conv2d_block(net, 128, 3, 1, self.train_flag, name='conv2_1a')
			self.layers['conv2'] = net  # 1/2 X
			net = conv2d_block(net, 128, 3, 2, self.train_flag, name='conv2_2a')

		with tf.variable_scope('conv3a'):
			net = conv2d_block(net, 256, 3, 1, self.train_flag, name='conv3_1a')
			self.layers['conv3'] = net  # 1/4 X
			# feat_map_img_1 = conv2d_block(net, 64, 3, 2, self.train_flag, name='conv3_2a')
			feat_map_img_1 = conv2d(net, 64, 3, 3, 2, 2, name='conv3_2a')
			self.probes['Img1FeatureMap'] = feat_map_img_1

		# Second Image Channel
		net = conv2d_block(self.input_img_2, 64, 3, 1, self.train_flag, name='conv1_1b')
		net = conv2d_block(net, 64, 3, 2, self.train_flag, name='conv1_2b')

		with tf.variable_scope('conv2b'):
			net = conv2d_block(net, 128, 3, 1, self.train_flag, name='conv2_1b')
			net = conv2d_block(net, 128, 3, 2, self.train_flag, name='conv2_2b')

		with tf.variable_scope('conv3b'):
			net = conv2d_block(net, 256, 3, 1, self.train_flag, name='conv3_1b')
			# feat_map_img_2 = conv2d_block(net, 64, 3, 2, self.train_flag, name='conv3_2b')
			feat_map_img_2 = conv2d(net, 64, 3, 3, 2, 2, name='conv3_2b')
			self.probes['Img2FeatureMap'] = feat_map_img_2

		with tf.variable_scope('correlation'):
			net, dims = correlation(feat_map_img_1, feat_map_img_2, self.correlation_param["kernel_size"],
							  self.correlation_param["max_displacement"],
							  self.correlation_param["stride_1"], self.correlation_param["stride_2"])
			self.probes['corr'] = net

			# Combine cross correlation results with convolution of RGB feature map
			c = 32
			feat_map_img_1_conv = conv2d_block(feat_map_img_1, c, 1, 1, self.train_flag, name='conv_redir')
			cc_relu = LeakyReLU(net)
			net = tf.concat((cc_relu, feat_map_img_1_conv), axis=3)
			dims = dims + c

		net = conv2d_block(net, 256, 3, 1, self.train_flag, name='conv3_1', last_dim=dims)
		self.layers['conv3_1'] = net  # 1/8 X

		net = conv2d_block(net, 512, 3, 2, self.train_flag, name='conv4_1')
		net = conv2d_block(net, 512, 3, 1, self.train_flag, name='conv4_2')
		self.layers['conv4'] = net  # 1/16 X

		net = conv2d_block(net, 512, 3, 2, self.train_flag, name='conv5_1')
		net = conv2d_block(net, 512, 3, 1, self.train_flag, name='conv5_2')
		self.layers['conv5'] = net  # 1/32 X
		return net

	@staticmethod
	def merge_shapes(feature_maps, name='reshape'):
		updated_feature_map = list()
		shape = None
		for idx, feature_map in enumerate(feature_maps):
			if idx == 0:
				shape = tf.shape(feature_map)
				updated_feature_map.append(feature_map)
			else:
				update_shape = (feature_map.shape[0], shape[1], shape[2], feature_map.shape[-1])
				updated_feature_map.append(tf.reshape(feature_map, shape=update_shape, name=name + '_' + str(idx)))
		tmp = updated_feature_map[0]
		updated_feature_map[0] = updated_feature_map[1]
		updated_feature_map[1] = tmp
		return updated_feature_map

	def build_refinement_stage(self):

		def deconv_func(scope_name, input_net, num_filters, prev_layer_input=None, multiplier=1):
			with tf.variable_scope(scope_name):
				flow = conv2d(input_net, 2, 3, 3, 1, 1, name='flow')
				deconv = deconv2d_block(input_net, int(num_filters/multiplier), 4, 2, name='deconv')
				up_flow = deconv2d(flow, 2, 4, 2, name='up_flow')
				if prev_layer_input is not None:
					in_list = self.merge_shapes((prev_layer_input, deconv, up_flow))
				else:
					in_list = self.merge_shapes((deconv, up_flow))
				out_net = tf.concat(in_list, axis=3)
				return out_net, flow

		# H/32 x W/32
		net = self.layers['conv5']
		net_in = self.layers['conv4']
		net, flow = deconv_func("Refine6", net, 512, prev_layer_input=net_in)
		self.layers['flow_level_6'] = flow  # 1/32 X

		# H/16 x W/16
		net_in = self.layers['conv3_1']
		net, flow = deconv_func("Refine7", net, 512, prev_layer_input=net_in)
		self.layers['flow_level_5'] = flow  # 1/16 X

		# H/8 x W/8
		net, flow = deconv_func("Refine8", net, 256, prev_layer_input=None)
		self.layers['flow_level_4'] = flow  # 1/8 X

		# H/4 x W/4
		net_in = self.layers['conv2']
		net, flow = deconv_func("Refine9", net, 256, prev_layer_input=net_in)
		self.layers['flow_level_3'] = flow  # 1/4 X
		self.probes['Refine9'] = net

		# H/2 x W/2
		net_in = self.layers['conv1']
		net, flow = deconv_func("Refine10", net, 256, prev_layer_input=net_in)
		self.layers['flow_level_2'] = flow  # 1/2 X

		net = conv2d(net, 2, 3, 3, 1, 1, name='conv1')
		self.layers['flow_level_1'] = net
		return net

	def add_conventional_post_process(self):
		with tf.variable_scope('ClassicPostProcess'):
			flow = self.layers['flow_level_1']
			self.layers['flow'] = flow * 20.0
			self.output_node_names.append(flow.op.name)

	def add_conventional_loss(self):
		with tf.variable_scope('ClassicLoss'):
			flow = self.target_img * 0.05

			losses = []
			# INPUT_HEIGHT, INPUT_WIDTH = float(flow.shape[1].value), float(flow.shape[2].value)

			# # L2 loss between predict_flow6, blob23 (weighted w/ 0.32)
			# predict_flow6 = self.layers['flow_level_6']
			# size = [tf.shape(predict_flow6)[1], tf.shape(predict_flow6)[2]]
			# downsampled_flow6 = tf.image.resize_bilinear(flow, tf.stack(size), align_corners=True, name='resize_f6')
			# losses.append(average_endpoint_error(downsampled_flow6, predict_flow6))
			#
			# # L2 loss between predict_flow5, blob28 (weighted w/ 0.08)
			# predict_flow5 = self.layers['flow_level_5']
			# size = [tf.shape(predict_flow5)[1], tf.shape(predict_flow5)[2]]
			# downsampled_flow5 = tf.image.resize_bilinear(flow, tf.stack(size), align_corners=True, name='resize_f5')
			# losses.append(average_endpoint_error(downsampled_flow5, predict_flow5))
			#
			# # L2 loss between predict_flow4, blob33 (weighted w/ 0.02)
			# predict_flow4 = self.layers['flow_level_4']
			# size = [tf.shape(predict_flow4)[1], tf.shape(predict_flow4)[2]]
			# downsampled_flow4 = tf.image.resize_bilinear(flow, tf.stack(size), align_corners=True, name='resize_f4')
			# losses.append(average_endpoint_error(downsampled_flow4, predict_flow4))
			#
			# # L2 loss between predict_flow3, blob38 (weighted w/ 0.01)
			# predict_flow3 = self.layers['flow_level_3']
			# size = [tf.shape(predict_flow3)[1], tf.shape(predict_flow3)[2]]
			# downsampled_flow3 = tf.image.resize_bilinear(flow, tf.stack(size), align_corners=True, name='resize_f3')
			# losses.append(average_endpoint_error(downsampled_flow3, predict_flow3))
			#
			# # L2 loss between predict_flow2, blob43 (weighted w/ 0.005)
			# predict_flow2 = self.layers['flow_level_2']
			# size = [tf.shape(predict_flow2)[1], tf.shape(predict_flow2)[2]]
			# downsampled_flow2 = tf.image.resize_bilinear(flow, tf.stack(size), align_corners=True, name='resize_f2')
			# losses.append(average_endpoint_error(downsampled_flow2, predict_flow2))

			# L1 loss between predict_flow2, blob43 (weighted w/ 0.005)
			predict_flow1 = self.layers['flow_level_1']
			size = [tf.shape(predict_flow1)[1], tf.shape(predict_flow1)[2]]
			# downsampled_flow2 = tf.image.resize_bilinear(flow, tf.stack(size), align_corners=True, name='resize_f1')
			ave_loss, orig_loss = average_endpoint_error(flow, predict_flow1)
			losses.append(ave_loss)
			self.probes['flow_level_1'] = predict_flow1
			self.probes['gt_flow'] = flow
			self.probes['orig_loss'] = orig_loss

			self.losses['level_losses'] = losses
			# loss = tf.losses.compute_weighted_loss(losses, [0.32, 0.08, 0.02, 0.01, 0.005, 0.8])
			loss = tf.losses.compute_weighted_loss(losses, [1])
			self.losses['weighted_loss'] = loss
			# Return the 'total' loss: loss fns + regularization terms defined in the model

			self.losses['total_loss'] = tf.reduce_mean(loss)

	def build_model(self):
		super().build_model()
		with tf.device(self._device):
			# net_input = self.add_image_normalise_layer(self.input_img)
			self.train_flag = tf.Variable(False, trainable=False, name='train_mode')
			self.build_correlation_stage()
			self.build_refinement_stage()
			if self.classic_mode:
				self.add_conventional_post_process()
				self.add_conventional_loss()
			else:
				# TODO: Implemented custom end and loss
				pass

			tf.summary.FileWriter(os.path.join('Graph'), tf.get_default_graph())


if __name__ == "__main__":
	flow_net = FlowNet()
	flow_net.build_model()
