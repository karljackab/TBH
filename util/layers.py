import tensorflow as tf

def fc_layer(name, bottom, output_dim, bias_term=True, weights_initializer=None,
             biases_initializer=None):
    # flatten bottom input
    # input has shape [batch, in_height, in_width, in_channels]
    shape = bottom.get_shape().as_list()
    input_dim = 1
    for d in shape[1:]:
        input_dim *= d
    flat_bottom = tf.reshape(bottom, [-1, input_dim])
    # weights and biases variables
    with tf.compat.v1.variable_scope(name):
        # initialize the variables
        if weights_initializer is None:
            weights_initializer = tf.random_normal_initializer(stddev=0.01)
        if bias_term and biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)

        # weights has shape [input_dim, output_dim]
        weights = tf.compat.v1.get_variable("kernel", [input_dim, output_dim], initializer=weights_initializer)
        if bias_term:
            biases = tf.compat.v1.get_variable("bias", output_dim, initializer=biases_initializer)
            fc = tf.compat.v1.nn.xw_plus_b(flat_bottom, weights, biases)
        else:
            fc = tf.matmul(flat_bottom, weights)
    return fc

def fc_relu_layer(name, bottom, output_dim, bias_term=True, weights_initializer=None, biases_initializer=None):
    fc = fc_layer(name, bottom, output_dim, bias_term, weights_initializer, biases_initializer)
    relu = tf.nn.relu(fc)
    return relu

OVERFLOW_MARGIN = 1e-8

def build_adjacency_hamming(tensor_in, code_length=32):
    """
    Build adjacency matrix by hamming distance
    Args:
        tensor_in: Hashing vectors
        code_length:
    Returns:
    """
    m1 = tensor_in - 1
    c1 = tf.matmul(tensor_in, m1, transpose_b=True)
    c2 = tf.matmul(m1, tensor_in, transpose_b=True)

    normalized_dist = tf.math.abs(c1 + c2) / code_length
    return tf.pow(1 - normalized_dist, 1.4)

def graph_laplacian(adjacency, size):
    """
    :param adjacency: must be self-connected
    :return: 
    """
    graph_size = size
    d = adjacency @ tf.ones([graph_size, 1])
    d_inv_sqrt = tf.pow(d + OVERFLOW_MARGIN, -0.5)
    d_inv_sqrt = tf.eye(graph_size) * d_inv_sqrt
    laplacian = d_inv_sqrt @ adjacency @ d_inv_sqrt
    return laplacian


def spectrum_conv_layer(name, tensor_in, adjacency, out_dim, size):
    """
    Convolution on a graph with graph Laplacian
    :param name:
    :param tensor_in: [N D]
    :param adjacency: [N N]
    :param out_dim:
    :return:
    """
    fc_sc = fc_layer(name, tensor_in, output_dim=out_dim)
    conv_sc = graph_laplacian(adjacency, size) @ fc_sc
    return conv_sc
