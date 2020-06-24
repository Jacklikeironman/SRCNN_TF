from utils import read_data,prepare_data,preprocess,imsave

import time
import random
import os
import matplotlib.pyplot as plt
'''
ubuntu只能使用不用GUI的，而matplotlib默认使用backend需要GUi，所以需要修改
'''
#plt.switch_backend('agg')
import numpy as np
import tensorflow as tf
import scipy.misc
import scipy.ndimage


class SRCNN(object):

    def __init__(self,sess,config):
        self.sess = sess
        self.mode = config.mode
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.image_size = config.image_size
        self.label_size = config.label_size
        self.c_dim = config.c_dim
        self.scale = config.scale
      
        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir
        self.log_dir = config.log_dir
        self.h5_train = config.h5_train
        self.h5_test = config.h5_test
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')

        self.weights = {
                'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
                'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
                'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
                }
        self.biases = {
                'b1': tf.Variable(tf.zeros([64]), name='b1'),
                'b2': tf.Variable(tf.zeros([32]), name='b2'),
                'b3': tf.Variable(tf.zeros([1]), name='b3')
                }
       
        self.predictions = self.inference()
        self.loss = tf.reduce_mean(tf.square(self.labels - self.predictions))
        self.metric = tf.multiply(10.0,tf.log(1.0 * 1.0 / self.loss)/tf.log(10.0))

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('PSNR', self.metric)
        self.merged_summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def inference(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
        return conv3
    
    def train(self):
        train_data,train_label = read_data(self.h5_train)
        test_data,test_label = read_data(self.h5_test)
        # print(train_data.shape)  #(21760, 33, 33, 1)
        # print(train_label.shape) #(21760, 21, 21, 1)
        self.summary_writer = tf.summary.FileWriter(self.log_dir,graph=tf.get_default_graph())
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        tf.global_variables_initializer().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if self.mode == 'train':
            print('************************Training***************************')
            for ep in range(self.epoch):
                batch_idxs = len(train_data) // self.batch_size
                for idx in range(0,batch_idxs):
                    batch_images = train_data[idx * self.batch_size : (idx + 1) * self.batch_size]
                    batch_labels = train_label[idx * self.batch_size : (idx + 1) * self.batch_size]

                    counter +=1
                    feed_dict = {self.images:batch_images,self.labels:batch_labels}
                    _,err,psnr = self.sess.run([self.train_op,self.loss,self.metric],feed_dict = feed_dict)
                    summary = self.sess.run(self.merged_summary_op,feed_dict = feed_dict)
                    self.summary_writer.add_summary(summary,counter)

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f], PSNR: [%.4f]" \
                             % ((ep+1), counter, time.time()-start_time, err, psnr))
                    if counter % 500 == 0:
                        x = random.choice(range(0,len(test_data) - self.batch_size))
                        sample_images,sample_labels = test_data[x:x + self.batch_size],test_label[x:x + self.batch_size]
                        feed_dict = {self.images: sample_images, self.labels: sample_labels}
                        err,psnr = self.sess.run([self.loss,self.metric],feed_dict = feed_dict)
                        print("[*]Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f], PSNR: [%.4f]" \
                              % ((ep + 1), counter, time.time() - start_time, err, psnr))
                        self.save(self.checkpoint_dir,counter)
        if self.mode == 'evaluate':
            print('************************Evaluating**************************')


    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step,
                        write_meta_graph=False)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
