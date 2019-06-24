import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import math


def Conv(name, x, filter_size, in_filters, out_filters, strides):

    with tf.variable_scope(name):
        n= 1 / np.sqrt(filter_size * filter_size * in_filters)
        kernel = tf.get_variable('filter', [filter_size, filter_size, in_filters, out_filters],tf.float32, initializer=tf.random_uniform_initializer(minval = -n, maxval = n))
        bias = tf.get_variable('bias',[out_filters],tf.float32, initializer=tf.random_uniform_initializer(minval = -n, maxval = n))
        
        return tf.nn.conv2d(x, kernel, [1,strides,strides,1], padding='SAME') + bias
    
def Channel_attention(name, x, ratio, n_feats):
    
    _res = x
    
    x = tf.reduce_mean(x, axis = [1,2], keep_dims = True)
    x = Conv(name + '_conv1', x, 1, n_feats, n_feats // ratio, 1)
    x = tf.nn.relu(x)
    
    x = Conv(name + '_conv2', x, 1, n_feats // ratio, n_feats, 1)
    x = tf.nn.sigmoid(x)
    x = tf.multiply(x, _res)
    
    return x

def RCA_Block(name, x, filter_size, ratio, n_feats):
    
    _res = x
    
    x = Conv(name + '_conv1', x, filter_size, n_feats, n_feats, 1)
    x = tf.nn.relu(x)
    x = Conv(name + '_conv2', x, filter_size, n_feats, n_feats, 1)
    
    x = Channel_attention(name + '_CA', x, ratio, n_feats)
    
    x = x + _res
    
    return x

def Residual_Group(name, x, n_RCAB, filter_size, ratio, n_feats):
    skip_connection = x
    
    for i in range(n_RCAB):
        x = RCA_Block(name + '_%02d_RCAB'%i, x, filter_size, ratio, n_feats)
    
    x = Conv(name + '_conv_last', x, filter_size, n_feats, n_feats, 1)
    
    x = x + skip_connection
    
    return x

def Up_scaling(name, x, kernel_size, n_feats, scale):
    
    ## if scale is 2^n
    if (scale & (scale -1) == 0):
        for i in range(int(math.log(scale, 2))):
            x = Conv(name + '_conv%02d'%i, x, kernel_size, n_feats, 2 * 2 * n_feats, 1)
            x = tf.depth_to_space(x, 2)
            
    elif scale == 3:
        x = Conv(name + '_conv', x, kernel_size, n_feats, 3 * 3 * n_feats, 1)
        x = tf.depth_to_space(x, 3)
        
    else:
        x = Conv(name + '_conv', x, kernel_size, scale * scale * n_feats, 1)
        x = tf.depth_to_space(x, scale)
        
    return x
        
def Mean_shifter(x, rgb_mean, sign, rgb_range = 255):
    
    kernel = tf.constant(name = "identity", shape = [1,1,3,3], value = np.eye(3).reshape(1,1,3,3), dtype = tf.float32)
    bias = tf.constant(name = 'bias', shape = [3], value = [ele * rgb_range * sign for ele in rgb_mean], dtype = tf.float32)
    
    return tf.nn.conv2d(x,kernel,[1,1,1,1],padding = "SAME") + bias    


_rgb_to_YCbCr_kernel = [[65.738 / 256 , -37.945 / 256, 112.439 / 256],
                       [129.057 / 256, -74.494 / 256, -94.154 / 256],
                       [25.064 / 256, 112.439 / 256, -18.214 / 256]]

def rgb_to_ycbcr(image):

    images = ops.convert_to_tensor(image, name='images_rgb')
    kernel = ops.convert_to_tensor(_rgb_to_YCbCr_kernel, dtype=images.dtype, name='ycbcr_kernel')
    
    ndims = images.get_shape().ndims

    img = math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])

    kernel = tf.constant(name = "identity", shape = [1,1,3,3], value = np.eye(3).reshape(1,1,3,3), dtype = tf.float32)

    bias = tf.constant(name = 'bias', shape = [3], value = [16.0, 128.0, 128.0], dtype = tf.float32)

    return tf.nn.conv2d(img ,kernel,[1,1,1,1],padding = "SAME") + bias


_YCbCr_to_rgb_kernel = [[298.082 / 256, 298.082 / 256, 298.082 / 256],
                       [0, -100.291 / 256, 516.412 / 256],
                       [408.583 / 256, -208.120 / 256, 0]]

def ycbcr_to_rgb(image):

    images = ops.convert_to_tensor(image, name='images')
    kernel = ops.convert_to_tensor(_YCbCr_to_rgb_kernel, dtype=images.dtype, name='rgb_kernel')
    
    ndims = images.get_shape().ndims

    img = math_ops.tensordot(images, kernel, axes=[[ndims - 1], [0]])

    kernel = tf.constant(name = "identity", shape = [1,1,3,3], value = np.eye(3).reshape(1,1,3,3), dtype = tf.float32)

    bias = tf.constant(name = 'bias', shape = [3], value = [-222.921, 135.576, -276.836], dtype = tf.float32)

    return tf.nn.conv2d(img ,kernel,[1,1,1,1],padding = "SAME") + bias


