import tensorflow as tf

from layers import *

def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    
    with tf.variable_scope('conv1') as scope:
        
        kernel = tf.get_variable('kernels', [3, 3, 1, 8], 
                            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [8],
                            initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(input, kernel, strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.relu(conv + biases, name=scope.name)

    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    
    with tf.variable_scope('conv2') as scope:
        
        kernel = tf.get_variable('kernels', [3, 3, 8, 8], 
                            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [8],
                            initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(conv1, kernel, strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.relu(conv + biases, name=scope.name)

    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    
    with tf.variable_scope('conv3') as scope:
        
        kernel = tf.get_variable('kernels', [3, 3, 8, 8], 
                            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [8],
                            initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(conv2, kernel, strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.relu(conv + biases, name=scope.name)

    # FC: output_dim: 100, no non-linearity
    with tf.variable_scope('fc') as scope:
        
        kernel = tf.get_variable('kernels', [4*4*8, 100], 
                            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [100],
                            initializer=tf.random_normal_initializer())
        fc = tf.matmul(tf.reshape(conv3, (-1, 128)), kernel) + biases

    return fc

def decoder(input):
    
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu

    with tf.variable_scope('fc_decode') as scope:
        
        kernel = tf.get_variable('kernels', [100, 128], 
                            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [128],
                            initializer=tf.random_normal_initializer())
        fc_de= tf.matmul(input, kernel) + biases
    
    # Reshape to [batch_size, 4, 4, 8]

    fc_de = tf.reshape(fc_de, [-1, 4, 4, 8])
    
    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu

    with tf.variable_scope('deconv1') as scope:
        
        kernel = tf.get_variable('kernels', [3, 3, 8, 8], 
                            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [8],
                            initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d_transpose(fc_de, kernel, [100, 8, 8, 8], strides=[1, 2, 2, 1], padding='SAME')
        deconv1 = tf.nn.relu(conv + biases, name=scope.name)
    
    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu

    with tf.variable_scope('deconv2') as scope:
        
        kernel = tf.get_variable('kernels', [8, 8, 1, 8], 
                            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [1],
                            initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d_transpose(deconv1, kernel, [100, 22, 22, 1], strides=[1, 2, 2, 1], padding='VALID')
        deconv2 = tf.nn.relu(conv + biases, name=scope.name)
    
    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    with tf.variable_scope('deconv3') as scope:
        
        kernel = tf.get_variable('kernels', [7, 7, 1, 1], 
                            initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases', [1],
                            initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d_transpose(deconv2, kernel, [100, 28, 28, 1], strides=[1, 1, 1, 1], padding='VALID')
        deconv3 = tf.nn.relu(conv + biases, name=scope.name)
    return deconv3

def autoencoder(input_shape):
    # Define place holder with input shape
    X = tf.placeholder(tf.float32, (None, 28, 28, 1))

    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        encoding = encoder(X)
        
        # Pass encoding into decoder to obtain reconstructed image
        decoding = decoder(encoding)
        
        # Return input image (placeholder) and reconstructed image
    return X, decoding
