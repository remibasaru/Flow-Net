import cv2

from util.data_handler_util import DataLoader
import tensorflow as tf
import os
import numpy as np


def load_graph(scope_name, graph_pb_path=os.path.join('Networks', 'net.pb')):
	with tf.gfile.GFile(graph_pb_path, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		with tf.device('/device:GPU:0'):
			tf.import_graph_def(graph_def, name=scope_name)

	return tf.get_default_graph()


class OpticalFlowDetector:

	def __init__(self):
		self.scope_name = "FlowNet"
		self.detection_graph = load_graph(self.scope_name)
		self.image_0 = self.detection_graph.get_tensor_by_name(self.scope_name + '/input_img_1:0')
		self.image_1 = self.detection_graph.get_tensor_by_name(self.scope_name + '/input_img_2:0')
		self.predicted_flow = self.detection_graph.get_tensor_by_name(self.scope_name + '/ClassicPostProcess/ResizeBilinear:0')
		# self.predicted_flow = self.detection_graph.get_tensor_by_name(self.scope_name + '/conv11/BiasAdd:0')
		self.sess = tf.Session(graph=self.detection_graph)

	@staticmethod
	def generate_pretty_flow(flow):
		flow = flow[0]
		flow = (flow + 10) * 10
		r_channel = np.zeros((flow.shape[0], flow.shape[1], 1), dtype=flow.dtype)
		flow = np.concatenate((flow, r_channel), axis=-1)
		flow = flow.astype(np.uint8)
		return flow

	def predict(self, img_blob, display=False):
		feed_dict = {
			self.image_0: img_blob["img_a"],
			self.image_1: img_blob["img_b"]
		}
		flow = self.sess.run(self.predicted_flow, feed_dict=feed_dict)
		if display:
			pretty_flow = self.generate_pretty_flow(flow)
			cv2.imshow('Result', pretty_flow)
			if 'gt_flow' in img_blob:
				gt_pretty_flow = self.generate_pretty_flow(img_blob['gt_flow'])
				cv2.imshow('GT', gt_pretty_flow)
			cv2.waitKey(0)


if __name__ == "__main__":
	optical_flow_detector = OpticalFlowDetector()

	data_loader = DataLoader(batch_size=1)
	data_loader.reset("VAL")
	while data_loader.is_next():
		blob = data_loader.get_next_batch()
		optical_flow_detector.predict(blob, True)

	# video_loader = VideoLoader()
	#
	# while video_loader.success:
	# 	video_loader.get_next_frame()
	# 	blob = {
	# 		"img_a": video_loader.frame_0,
	# 		"img_b": video_loader.frame_1
	# 	}
	# 	optical_flow_detector.predict(blob)
