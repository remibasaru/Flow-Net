# --------------------------------------------------------
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------
import errno
import os
import pickle
import random

from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re

import time


class Timer(object):
	"""A simple timer."""

	def __init__(self):
		self.total_time = 0.
		self.calls = 0
		self.start_time = 0.
		self.diff = 0.
		self.average_time = 0.

	def tic(self):
		# using time.time instead of time.clock because time time.clock
		# does not normalize for multithreading
		self.start_time = time.time()

	def toc(self, average=True):
		self.diff = time.time() - self.start_time
		self.total_time += self.diff
		self.calls += 1
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		else:
			return self.diff


class Trainer(object):

	def __init__(self, _expDir):
		self.saver = None
		self.expDir = _expDir
		self.model_figure_path = None

	def update_opts(self):
		pass

	def model_path(self, ep):
		return os.path.join(self.expDir, 'net-epoch-' + str(ep), 'model.ckpt')

	def model_folder_path(self, ep):
		return os.path.join(self.expDir, 'net-epoch-' + str(ep))

	@staticmethod
	def get_variables_in_checkpoint_file(file_name):
		try:
			reader = pywrap_tensorflow.NewCheckpointReader(file_name)
			var_to_shape_map = reader.get_variable_to_shape_map()
			return var_to_shape_map
		except Exception as e:  # pylint: disable=broad-except
			print(str(e))
			if "corrupted compressed block contents" in str(e):
				print("It's likely that your checkpoint file has been compressed "
					  "with SNAPPY.")

	def load_state(self, prev_pos_1, sess):
		model_folder_path = self.model_folder_path(prev_pos_1)
		model_path = self.model_path(prev_pos_1)
		self.saver.restore(sess, model_path)
		stats = None
		with open(os.path.join(model_folder_path, 'stats.pickle'), 'rb') as handle:
			stats = pickle.load(handle)

		return stats, sess

	def save_state(self, epoch, state):
		save_path_folder = self.model_folder_path(epoch)
		save_path = self.model_path(epoch)
		try:
			os.mkdir(save_path_folder)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
			pass
		_sess = state['sess']
		save_path = self.saver.save(_sess, save_path)
		print("Model saved in path: %s" % save_path)
		return True

	def save_stats(self, epoch, stats):
		stats_path = self.model_folder_path(epoch)
		with open(os.path.join(stats_path, 'stats.pickle'), 'wb') as handle:
			pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
		return None

	def find_last_checkpoint(self):
		epoch = 0
		for f in os.listdir(os.path.join('.', self.expDir)):
			if re.match(r'net-epoch-\d+', f):
				tmp_epoch = int((re.search(r'\d+', f)).group(0))
				if tmp_epoch > epoch:
					epoch = tmp_epoch
		return epoch

	def plot_stats(self, fig, stats):

		fig.clf()
		train_objective, val_objective, epoch_count = transpose_stats(stats)

		num_var = len(train_objective)
		dim = np.ceil(num_var ** .5)
		dim = int(dim * 100 + dim * 10)
		idx = np.arange(epoch_count).astype(np.int) + 1

		for i, vars in enumerate(train_objective.keys()):
			plt.subplot(dim + i + 1)

			plt.plot(idx, val_objective[vars], 'ro-', label='val')
			plt.plot(idx, train_objective[vars], 'bo-', label='train')

			plt.title(vars)
			plt.xlabel('epoch')
			plt.grid(True)
			plt.legend()

		plt.show(block=False)
		fig.canvas.draw()

		plt.savefig(self.model_figure_path)
		return None

	def process_epoch(self, sess, state, params, timer, mode):
		raise NotImplementedError

	def train(self, sess, opts):
		timer = Timer()
		if opts['plotStatistics']:
			# TO DO: Fix lack of figure update once figure is out of focus
			fig = plt.gcf()
			fig.show()
			fig.canvas.draw()

		if opts['continue'] is not None:
			prev_pos_1 = max(0, min(opts['continue'], self.find_last_checkpoint()))
		else:
			prev_pos_1 = max(0, self.find_last_checkpoint())

		start_1 = prev_pos_1 + 1
		if prev_pos_1 >= 1:
			print('Resuming by loading epoch', str(prev_pos_1))
			stats, sess = self.load_state(prev_pos_1, sess)

			if sess is None or stats is None:
				stats = dict()
				stats['train'] = []
				stats['val'] = []
				print('Failed to load. Starting with epoch ', str(start_1), '\n')
			else:
				print('Continuing at epoch ', str(start_1), '\n')
		else:
			stats = dict()
			stats['train'] = []
			stats['val'] = []
			print('Starting at epoch ', str(start_1), '\n')
		state = dict()
		opts['session'] = sess

		for ep in range(start_1 - 1, opts['numEpochs']):
			# Set the random seed based on the epoch and opts.randomSeed.
			# This is important for reproducibility, including when training
			# is restarted from a checkpoint.
			epoch = ep + 1
			random.seed(epoch + opts['randomSeed'])

			# Train for one epoch
			params = opts
			params['epoch'] = epoch

			state = self.process_epoch(sess, state, params, timer, 'TRAIN')
			state = self.process_epoch(sess, state, params, timer, 'VAL')

			self.save_state(epoch, state)
			last_stats = state['stats']

			stats['train'].append(last_stats['train'])
			stats['val'].append(last_stats['val'])

			self.save_stats(epoch, stats)

			if opts['plotStatistics']:
				self.plot_stats(fig, stats)
		return sess, stats

	def update_session_with_optimum_epoch_state(self, modelDir, sess, loss_name='total_loss'):

		last_epoch_id = self.find_last_checkpoint()

		epoch_stats_path = os.path.join(modelDir, 'net-epoch-' + str(last_epoch_id), 'stats.pickle')
		epoch_stats = pickle.load(open(epoch_stats_path, "rb"))
		min_loss = np.Inf
		min_loss_ep_id = -1
		for ep_id in range(last_epoch_id):
			ep_loss_ave = epoch_stats['val'][ep_id][loss_name]['average']
			if ep_loss_ave < min_loss:
				min_loss = ep_loss_ave
				min_loss_ep_id = ep_id
		# min_loss_ep_path = os.path.join(modelDir, 'net-epoch-' + str(min_loss_ep_id + 1))
		if min_loss_ep_id >= 0:
			_, sess = self.load_state(min_loss_ep_id + 1, sess)
		return sess


def transpose_stats(stats):
	train_stats = stats['train']
	val_stats = stats['val']
	train_objective = {}
	val_objective = {}
	epoch_count = len(val_stats)

	for idx, train_stat in enumerate(train_stats):
		val_stat = val_stats[idx]
		for objective in train_stat.keys():
			if objective not in train_objective:
				train_objective[objective] = []
			if objective not in val_objective:
				val_objective[objective] = []
			train_objective[objective].append(train_stat[objective]['average'])
			val_objective[objective].append(val_stat[objective]['average'])

	return train_objective, val_objective, epoch_count
