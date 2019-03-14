import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
FULL_CONNECTED_NODE = 1024

# convolution
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# weight initialization
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)


# CNN 28x28 -> conv5x5 = 28x28x32-> pool2x2 = 14x14x32
#      -> conv5x5 = 14x14x64 -> pool2x2 = 7x7x64
#      -> densely = 1024x1
def inference(input_tensor, keep_prob):
    # if try to get variable twice in same scope with same name, set reuse = tf.AUTO_REUSE
    with tf.variable_scope('layer1'):
        # first convolutinal layer
        w_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        x_image = tf.reshape(input_tensor, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('layer2'): 
        # second convolutional layer
        w_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.variable_scope('layer3'):
        # densely connected layer
        w_fc1 = weight_variable([7*7*64, FULL_CONNECTED_NODE])
        b_fc1 = bias_variable([FULL_CONNECTED_NODE])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        
        # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
    with tf.variable_scope('layer4'):
        # readout layer
        w_fc2 = weight_variable([FULL_CONNECTED_NODE, OUTPUT_NODE])
        b_fc2 = bias_variable([OUTPUT_NODE])
        # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
        y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    return y_conv