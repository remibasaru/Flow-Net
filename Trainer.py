# --------------------------------------------------------
# Written by Rilwan Remilkeun Basaru
# --------------------------------------------------------

import os
import tensorflow as tf

from model.flow_net_util import FlowNet
from util.data_handler_util import DataLoader
from util.trainer_util import Trainer


class FlowTrainer(Trainer):
    def __init__(self, param):

        super().__init__(param['expDir'])
        if param['gpu']:
            self._device = "/device:GPU:0"
        else:
            self._device = "/cpu:0"
        self.opts = param
        if not param['debug']:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if not os.path.isdir(param['graphPathTrain']):
            os.mkdir(param['graphPathTrain'])
        if not os.path.isdir(param['graphPathTest']):
            os.mkdir(param['graphPathTest'])
        if not os.path.isdir(param['outputModelPath']):
            os.mkdir(param['outputModelPath'])
        if not os.path.isdir(param['expDir']):
            os.mkdir(param['expDir'])

        self.model_figure_path = os.path.join(param['expDir'], 'net-train.pdf')

        self.graph_path = param['graphPathTrain']
        self.net = FlowNet(graph_path_train=param['graphPathTrain'], graph_path_pred=param['graphPathTest'],
                               device=self._device, batch_size=param['batchSize'])

        # debug probes
        self._debug_probes = {}
        self.data_loader = DataLoader(os.path.join('Data', 'train'), batch_size=param['batchSize'])

        self.trainable_graph = tf.Graph()
        self.frozen_graph = None

    def setup(self, frozen=False):
        # Create session
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        if not frozen:
            sess = tf.Session(config=tfconfig, graph=self.trainable_graph)
        else:
            sess = tf.Session(config=tfconfig, graph=self.frozen_graph)
        with sess.graph.as_default():
            self.net.build_model()
            self.net.add_optimizer(self.opts)
            variables = tf.global_variables()
            # Initialize all variables first
            sess.run(tf.variables_initializer(variables))
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            self.saver = self.net.saver = tf.train.Saver()

        return sess

    def process_epoch(self, sess, state, params, timer,  mode):

        epoch = params['epoch']
        stats = dict()

        if not state:
            state['stats'] = dict()

        losses_name = ["total_loss"]
        for l in losses_name:
            stats[l] = {'count': 0, 'average': 0}
        ite = 1
        self.data_loader.reset(mode)
        while self.data_loader.is_next():

            timer.tic()
            # Get training data, one batch at a time
            blobs = self.data_loader.get_next_batch()

            # Compute the graph without summary
            stats = self.net.train_step(sess, blobs, stats, mode)
            # stats = self.net.test_step_and_display_result(sess, blobs, stats)
            # stats = self.net.test_step_and_probe(sess, blobs, stats)

            timer.toc()

            # Display training information
            print('%s: epoch %d:\t %d/%d: (%.2fs)\t cross_entropy_loss: %.6f ' %
                  (mode.lower(), epoch, ite, self.data_loader.ite_count(), timer.average_time,
                   stats["total_loss"]['average']))
            ite = ite + 1
        # Save back to state
        state['stats'][mode.lower()] = stats
        state['sess'] = sess
        return state

    def trainClassifier(self, sess):
        sess, stats = super().train(sess, self.opts)
        return sess, stats
