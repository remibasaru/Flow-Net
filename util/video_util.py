import cv2
import numpy as np


class VideoLoader:
	def __init__(self, video_file_path):
		path = video_file_path
		self.vidcap = cv2.VideoCapture(path)
		self.count = 0
		self.success = True
		self.frame_0 = None
		self.frame_1 = None
		self.get_next_frame()

	@staticmethod
	def convert2rgb(bgr_im, target_shape=None):
		assert bgr_im.shape[-1] == 3, 'color chanel should be in the last axis'
		_axis = len(bgr_im.shape) - 1
		tmp_im = np.split(bgr_im, 3, axis=_axis)
		rgb_im = np.squeeze(np.stack((tmp_im[2], tmp_im[1], tmp_im[0]), axis=_axis))
		rgb_im = rgb_im[:, :512, :]
		if target_shape is not None:
			rgb_im = cv2.resize(rgb_im, (target_shape[1], target_shape[0]))
		rgb_im = np.expand_dims(rgb_im, axis=0)

		return rgb_im.astype(np.float32)

	def get_next_frame(self, target_shape=(384, 512)):

		self.success, image = self.vidcap.read()
		self.frame_0 = self.frame_1
		self.frame_1 = self.convert2rgb(image, target_shape)
		self.count += 1



