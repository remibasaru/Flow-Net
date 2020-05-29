# --------------------------------------------------------
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------


import os
import random
import numpy as np

from util.flying_chair_dataset_util import read


class DataLoader:
	def __init__(self, data_path, batch_size):
		#  Download from Flying Chair Dataset from https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs
		self.data_path = data_path
		self.length = 8000
		# self.length = 4
		self.indices = random.sample(range(20000), self.length)
		# self.length = 40
		# self.indices = range(self.length)

		self.data_indices = None
		self.ite = 0
		self.batch_size = batch_size

	def _next_img(self, idx):
		prefix = '%07d' % idx
		print(os.path.join(self.data_path, prefix))
		img_a_path = os.path.join(self.data_path, prefix + "-img_0.png")
		img_b_path = os.path.join(self.data_path, prefix + "-img_1.png")
		gt_flow_path = os.path.join(self.data_path, prefix + "-flow_01.flo")
		gt_flow = read(gt_flow_path).astype(np.float32)
		img_a = read(img_a_path).astype(np.float32)
		img_b = read(img_b_path).astype(np.float32)
		return img_a, img_b, gt_flow

	def get_blob(self, indices):
		# idx = self.data_indices[self.ite]
		max_shape = [384, 512]
		num_images = len(indices)
		images_a = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
		images_b = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
		gt_flows = np.zeros((num_images, max_shape[0], max_shape[1], 2), dtype=np.float32)

		for i in range(num_images):
			img_a, img_b, gt_flow = self._next_img(indices[i])
			images_a[i, :, :, :] = img_a
			images_b[i, :, :, :] = img_b
			gt_flows[i, :, :, :] = gt_flow
		blob = {
			"img_a": images_a,
			"img_b": images_b,
			"gt_flow": gt_flows
		}
		return blob

	def get_next_batch(self):
		db_inds = self.data_indices[self.ite: min(len(self.data_indices), self.ite + self.batch_size)]
		blob = self.get_blob(db_inds)
		self.ite += self.batch_size
		return blob

	def reset(self, mode):
		train_count = round(0.8 * float(len(self.indices)))
		if mode.lower() == 'train':
			self.data_indices = self.indices[:train_count]
		else:
			self.data_indices = self.indices[:train_count]
			# self.data_indices = self.indices[train_count:]
		self.ite = 0

	def ite_count(self):
		return np.floor(len(self.data_indices) / self.batch_size)

	def is_next(self):
		if self.ite >= len(self.data_indices):
			return False
		else:
			return True


if __name__ == "__main__":
	data_loader = DataLoader(os.path.join('Data', 'train'), 5)
	data_loader.reset('TRAIN')
	data_loader.get_next_batch()
