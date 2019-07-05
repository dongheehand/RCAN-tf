import tensorflow as tf
from ops import *
import numpy as np
from data_loader import dataloader


class RCA_net():
    
    def __init__(self, args):
        
        self.data_loader = dataloader(args)
        
        self.channel = args.channel
        self.scale = args.scale
        self.n_feats = args.n_feats
        self.n_RG = args.n_RG
        self.n_RCAB = args.n_RCAB
        self.kernel_size = args.kernel_size
        self.ratio = args.ratio
        self.in_memory = args.in_memory
        self.learning_rate = args.learning_rate
        self.decay_step = args.decay_step
        self.decay_rate = args.decay_rate
        self.mode = args.mode
        self.batch_size = args.batch_size
        self.rgb_mean = [0.4488, 0.4371, 0.4040]
        
        
    def build_graph(self):
        
        ################## Input ####################################
        
        if self.in_memory:
            self.LR = tf.placeholder(name = "LR", shape = [None, None, None, self.channel], dtype = tf.float32)
            self.GT = tf.placeholder(name = "GT", shape = [None, None, None, self.channel], dtype = tf.float32)
            
            x = self.LR
            self.label = self.GT
        
        else:
            self.data_loader.build_loader()
            
            if self.mode == 'test_only':
                x = self.data_loader.next_batch
                self.label = tf.placeholder(name = 'dummy', shape = [None, None, None, self.channel], dtype = tf.float32)

            elif self.mode == 'train' or self.mode == 'test':
                x = self.data_loader.next_batch[0]
                self.label = self.data_loader.next_batch[1]
        
        ############################################################
        
        self.global_step = tf.placeholder(name = 'learning_step', shape = None, dtype = tf.int32)
        
        ################## Model ####################################
        
        x = Mean_shifter(x, self.rgb_mean, sign = -1, rgb_range = 255)
        
        x = Conv(name = "conv_SF", x = x, filter_size = self.kernel_size, in_filters = self.channel, out_filters = self.n_feats, strides = 1)
        LongSkipConnection = x
        
        for i in range(self.n_RG):
            x = Residual_Group('RG%02d'%i, x, self.n_RCAB, self.kernel_size, self.ratio, self.n_feats)
        
        x = Conv(name = 'conv_LSC', x = x, filter_size = self.kernel_size, in_filters = self.n_feats, out_filters = self.n_feats, strides = 1)
        x = x + LongSkipConnection
        
        x = Up_scaling('up_sample', x, self.kernel_size, self.n_feats, self.scale)
        x = Conv('conv_rec', x, self.kernel_size, self.n_feats, self.channel, 1)
            
        self.output = Mean_shifter(x, self.rgb_mean, sign = 1, rgb_range = 255)        
        
        self.loss = tf.reduce_mean(tf.abs(self.label - self.output))
        
        ############################################################
        
        ############## Optimizer ###################################
        
        self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_step, self.decay_rate ,staircase = True)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train = optimizer.minimize(self.loss)
        
        ############################################################
        
        ############## Logging & Outputs ###########################
        
        self.output = tf.clip_by_value(self.output, 0.0, 255.0)
        self.output = tf.round(self.output)
        self.output = tf.cast(self.output, tf.uint8)
        self.RGB_PSNR = tf.reduce_mean(tf.image.psnr(self.output, tf.cast(self.label, tf.uint8), max_val = 255))
        logging_loss = tf.summary.scalar(name = 'train_loss', tensor = self.loss)
        logging_RGB_PSNR = tf.summary.scalar(name = 'train_RGB_PSNR', tensor = self.RGB_PSNR)
        
        ############################################################
        
