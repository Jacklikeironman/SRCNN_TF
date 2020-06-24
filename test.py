# -*- coding: utf-8 -*-
from model import SRCNN
 
import matplotlib.pyplot as plt
import tensorflow as tf
import pprint
import os

flags = tf.app.flags
flags.DEFINE_string("mode", 'train', "run mode train or evaluate[train]")
flags.DEFINE_integer('batch_size',128,'Number of batch_size[128]')
flags.DEFINE_integer('epoch',200,'total epoch to run[200]')
flags.DEFINE_float('learning_rate',1e-4,'Learning rate[1e-3]')
flags.DEFINE_integer('image_size',33,'shape of input image[33]')
flags.DEFINE_integer('label_size',21,'shape of label[21]')
flags.DEFINE_integer('c_dim',1,'Number of channel[1]')
flags.DEFINE_integer('scale',3,'Default scale[3]')
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("log_dir", 'Log', "Name of log directory[Log]")
flags.DEFINE_string("h5_train", 'train.h5', "Name of  train h5 file [train.h5]")
flags.DEFINE_string("h5_test", 'test.h5', "Name of test h5 file [test.h5]")
FLAGS =flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    with tf.Session() as sess:
        srcnn = SRCNN(sess,FLAGS)
        srcnn.train()

if __name__ == '__main__':
    tf.app.run()