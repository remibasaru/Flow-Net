import os

import cv2
import tensorflow as tf
import numpy as np
import scipy.io


class Network:

	def __init__(self, graph_path_train, graph_path_pred, device="/cpu:0"):
		self._device = device
		self.validity_score = None
		self.output_node_names = []
		self.pred_tensors_t = {}
		self._graph_path_train = graph_path_train
		self._graph_path_pred = graph_path_pred
		self.num_classes = 3

		self.input_img_1 = None
		self.input_img_2 = None
		self.target_img = None

		self.mean_pixel_val = np.array([0.0, 0.0, 0.0])
		self.std_pixel_val = np.array([1.0, 1.0, 1.0])
		self.train_op = None
		self.train_flag = None
		self.probes = {}
		self.layers = {}
		self.losses = {}

	def handle_admin(self):
		w = None
		h = None
		bs = None
		self.input_img_1 = tf.placeholder(dtype=tf.float32, shape=[bs, h, w, 3], name="input_img_1")
		self.input_img_2 = tf.placeholder(dtype=tf.float32, shape=[bs, h, w, 3], name="input_img_2")
		self.target_img = tf.placeholder(dtype=tf.float32, shape=[bs, h, w, 2], name="target_img")

	def add_image_normalise_layer(self, image):
		with tf.variable_scope("normalize"):
			mean_pixel_val = tf.constant(self.mean_pixel_val, dtype=tf.float32, name='mean_pixel_val')
			std_pixel_val = tf.constant(self.std_pixel_val, dtype=tf.float32, name='mean_pixel_val')
			image -= mean_pixel_val
			image /= std_pixel_val
			return image

	def build_model(self):
		self.handle_admin()

	def add_loss(self):
		with tf.variable_scope("loss"):
			entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_img,
																	  logits=self.layers['output_logit'])

			mean_entropy_loss = tf.reduce_mean(entropy_loss)
			self.layers['entropy_loss'] = entropy_loss
			self.losses['cross_entropy_loss'] = mean_entropy_loss

	def add_optimizer(self, opts):
		lr = opts['learningRate']
		loss = self.losses['total_loss']

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			optimizer = tf.train.AdamOptimizer(learning_rate=lr)

			gvs = optimizer.compute_gradients(loss)
			# final_gvs = []
			#
			# with tf.variable_scope('Disc_filter'):
			# 	for grad, var in gvs:
			# 		if "correlation" in var.name or "conv1_1b" in var.name or "conv1_2b" in var.name or "conv2b" in var.name or "conv3b" in var.name:
			# 			grad = tf.multiply(grad, 0)
			# 		final_gvs.append((grad, var))
			# self.train_op = optimizer.apply_gradients(final_gvs)

			self.train_op = optimizer.apply_gradients(gvs)

	def _add_optimizer(self, opts, _type="Adam"):
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

		with tf.control_dependencies(update_ops):
			if _type == "Adam":
				optimizer = tf.train.AdamOptimizer(learning_rate=opts['learningRate'])
			elif _type == "RMS":
				optimizer = tf.train.RMSPropOptimizer(learning_rate=opts['learningRate'], decay=opts['weightDecay'])
			elif _type == "Momentum":
				optimizer = tf.train.MomentumOptimizer(learning_rate=opts['learningRate'], momentum=opts['momentum'])
			else:
				raise ValueError("Unknown optimizer")
			gvs = optimizer.compute_gradients(self.losses['total_loss'])

			self.train_op = optimizer.apply_gradients(gvs)

	def write_graph_to_tensorboard(self, sess, mode="pred"):
		if mode == "pred":
			tf.summary.FileWriter(self._graph_path_pred, sess.graph)
		else:
			tf.summary.FileWriter(self._graph_path_train, sess.graph)

	def update_stats(self, q_names, out_measures, stats):

		for idx, name in enumerate(q_names):
			loss_val = np.mean(out_measures[idx])
			cur_stats = stats[name]
			tot_loss = cur_stats['average'] * cur_stats['count'] + loss_val
			cur_stats['count'] = cur_stats['count'] + 1
			cur_stats['average'] = tot_loss / cur_stats['count']
			stats[name] = cur_stats
		return stats

	def extract_loss_from_layer(self, stats):
		q_set = []
		q_names = []
		for name in stats.keys():
			q_set.append(self.losses[name])
			q_names.append(name)
		return q_set, q_names

	def train_step(self, sess, blobs, stats, mode):
		feed_dict = {self.input_img_1: blobs["img_a"], self.input_img_2: blobs["img_b"],
					 self.target_img: blobs["gt_flow"], self.train_flag: False}

		if mode == 'TRAIN':
			# Run optimization op (back propagation) only in the training phase
			feed_dict[self.train_flag] = True
			sess.run([self.train_op], feed_dict=feed_dict)

		q_set, q_names = self.extract_loss_from_layer(stats)
		feed_dict[self.train_flag] = False

		# q_set.append(self.layers['soft'])
		out_measures = sess.run(q_set, feed_dict=feed_dict)
		# thresh_cost = self.test_threshold(out_measures.pop(), blobs["output_img"])
		stats = self.update_stats(q_names, out_measures, stats)

		return stats

	@staticmethod
	def display_image(img_rgb, pred_class_str, gt_class_str):
		tmp_im = np.split(img_rgb.astype(np.uint8), 3, axis=2)
		im_bgr = np.squeeze(np.stack((tmp_im[2], tmp_im[1], tmp_im[0]), axis=2))
		cv2.putText(im_bgr, "Pred: " + pred_class_str, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
		cv2.putText(im_bgr, "GT: " + gt_class_str, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
		cv2.imshow("Predicted Results", im_bgr)
		cv2.waitKey(0)

	@staticmethod
	def generate_pretty_flow(flow):
		flow = flow[0]
		flow = (flow + 10) * 10
		r_channel = np.zeros((flow.shape[0], flow.shape[1], 1), dtype=flow.dtype)
		flow = np.concatenate((flow, r_channel), axis=-1)
		flow = flow.astype(np.uint8)
		return flow

	def test_step_and_display_result(self, sess, blobs, stats, display=True):
		feed_dict = {self.input_img_1: blobs["img_a"], self.input_img_2: blobs["img_b"],
					 self.target_img: blobs["gt_flow"], self.train_flag: False}

		# q_set.append(self.layers['soft'])

		# flow = sess.run(self.layers['flow'], feed_dict=feed_dict)
		flow = sess.run(self.layers['flow_level_2'], feed_dict=feed_dict)

		if display:
			pretty_flow = self.generate_pretty_flow(flow)
			cv2.imshow('Result', pretty_flow)
			gt_pretty_flow = self.generate_pretty_flow(blobs['gt_flow'])
			cv2.imshow('GT', gt_pretty_flow)
			cv2.waitKey(0)
		return stats

	@staticmethod
	def display_tensor(tensor, compund=True):
		if (tensor.shape) == 3:
			tensor = np.expand_dims(tensor, 0)

		def convert_to_displayable(img):
			_range = img.max() - img.min()
			img = ((img - img.min())/_range) * 255
			img = img.astype(np.uint8)
			return img

		def display(im):
			cv2.imshow('Tensor', im)
			cv2.waitKey(0)

		tensor = convert_to_displayable(tensor[0])
		if compund:
			c = tensor.shape[-1]
			c_1 = int(np.ceil(np.power(c, .5)))
			img = np.zeros((tensor.shape[0], tensor.shape[1], int(np.power(c_1, 2))))
			img[:, :, :c] = tensor
			tensor = np.reshape(img, (tensor.shape[0], tensor.shape[1], c_1, c_1))
			tensor = np.transpose(tensor, (0, 2, 1, 3))
			tensor = np.reshape(tensor, (tensor.shape[0] * c_1, tensor.shape[2] * c_1))
			display(tensor)

		else:
			for i in range(tensor.shape[-1]):
				img = tensor[:, :, i]



	def test_step_and_probe(self, sess, blobs, stats):
		feed_dict = {self.input_img_1: blobs["img_a"], self.input_img_2: blobs["img_b"],
					 self.target_img: blobs["gt_flow"], self.train_flag: False}

		tensor_list = []
		item_names = []
		for item in self.probes.keys():
			item_names.append(item)
			tensor_list.append(self.probes[item])

		eval_list = sess.run(tensor_list, feed_dict=feed_dict)
		probe_blob = dict(zip(item_names, eval_list))
		scipy.io.savemat(os.path.join('Probe', 'matlab.mat'), mdict=probe_blob)
		return stats

	def freeze_net(self, sess):
		# Specify the real node name
		graph = sess.graph  # tf.get_default_graph()

		sess.run(tf.assign(self.train_flag, False))
		input_graph_def = graph.as_graph_def()

		# input_graph_def = Networks.fix_bnorm_bugs(input_graph_def)

		output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, self.output_node_names)

		return output_graph_def


if __name__ == "__main__":
	input_img = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3], name="input_img")
	train_flag = tf.Variable(False, trainable=False, name='train_mode')


