import tensorflow as tf

# Create AlexNet model
def conv1st(name, l_input, w, b):
    cov = tf.nn.conv2d(l_input, w, strides=[1, 4, 4, 1], padding='VALID')
    return tf.nn.relu(tf.nn.bias_add(cov,b), name=name)
    
def conv2d(name, l_input, w, b):
    cov = tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(cov,b), name=name)

def max_pool(name, l_input, k, s):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _dropout, n_classes, imagesize, img_channel):
    # Store layers weight & bias
    _weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, img_channel, 64])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256])),
        #'wd1': tf.Variable(tf.random_normal([8*8*256, 1024])),
        'wd1': tf.Variable(tf.random_normal([6*6*256, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([4096, n_classes]))
    }
    
    _biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([192])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([256])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([4096])),
        'bd2': tf.Variable(tf.random_normal([4096])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    # Reshape input picture OH WAIT NOPE CUS JE SUIS UN TENSAI DESU
    _X = tf.reshape(_X, shape=[-1, imagesize, imagesize, img_channel])

    # Convolution Layer
    conv1 = conv1st('conv1', _X, _weights['wc1'], _biases['bc1'])

    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=3, s=2)
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=3, s=2)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    conv4 = conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])
    conv5 = conv2d('conv5', conv4, _weights['wc5'], _biases['bc5'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv5, k=3, s=2)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out
