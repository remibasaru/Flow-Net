# --------------------------------------------------------
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------

from __future__ import print_function
from Trainer import FlowTrainer
import os
import tensorflow as tf


if __name__ == "__main__":

    opts = {}
    param = dict()
    param['filename'] = os.path.join('.', 'Networks', 'net.pb')

    opts['plotStatistics'] = True

    opts['expDir'] = "Exp"
    opts['gpu'] = True
    opts['batchSize'] = 1
    opts['numEpochs'] = 6

    opts['learningRate'] = 1e-5
    opts['weightDecay'] = 0.005
    opts['momentum'] = 0.09

    opts['continue'] = None  # set which epoch to continue from. Set to None to be ignored.
    opts['graphPathTrain'] = os.path.join('.', "Graph", 'TrainModel')
    opts['graphPathTest'] = os.path.join('.', "Graph", 'PredictionModel')
    opts['outputModelPath'] = os.path.join('.',  "Networks")
    # opts['gpu'] = opts['gpu']
    opts['randomSeed'] = 0

    trainer = FlowTrainer(opts)
    # Generate a trainable graph and return a session (a connection) to the graph
    trainable_graph_sess = trainer.setup()
    # Use returned session to train graph and store persistent data about graph
    trainable_graph_sess, training_stats = trainer.trainClassifier(trainable_graph_sess)

    trainable_graph_sess = trainer.update_session_with_optimum_epoch_state(opts['expDir'], trainable_graph_sess)
    output_graph_def = trainer.net.freeze_net(trainable_graph_sess)
    with tf.gfile.GFile(param['filename'], "wb") as f:
        f.write(output_graph_def.SerializeToString())

    # Close session
    trainable_graph_sess.close()
