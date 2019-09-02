# -*-coding:utf-8-*-
import tensorflow as tf


def conv_op(input_op, name, kh, kw, n_out, dh, dw):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.compat.v1.get_variable(scope + "w",
                                 shape=[kh, kw, n_in, n_out], dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding="SAME")
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)

        return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool2d(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding="SAME", name=name)


def fc_op(input_op, name, n_out):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.compat.v1.get_variable(scope + "w",
                                 shape=[n_in, n_out], dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.compat.v1.nn.relu_layer(input_op, kernel, biases, name=scope)

        return activation


def inference_op(images, n_classes):
    conv1_1 = conv_op(images, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)

    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
    conv3_3 = conv_op(conv3_2, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2) 

    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_3 = conv_op(conv4_2, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_3 = conv_op(conv5_2, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dh=2, dw=2)

    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    fc6 = fc_op(resh1, name="fc6", n_out=1024)

    with tf.compat.v1.variable_scope("softmax_linear") as scope:
        weights = tf.compat.v1.get_variable("weights",
                                  shape=[1024, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.compat.v1.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc6, weights), biases, name="softmax_linear")
    tf.compat.v1.add_to_collection("output", softmax_linear)

    return softmax_linear


def losses(logits, labels):
    with tf.compat.v1.variable_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
    return loss


def evaluation(logits, labels):
    with tf.compat.v1.variable_scope("accuracy"):
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    return accuracy
