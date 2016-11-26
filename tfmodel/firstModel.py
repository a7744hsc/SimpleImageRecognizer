import random

import numpy as np
import tensorflow as tf
import preprocess
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# from tensorflow.contrib.session_bundle import exporter
# from tensorflow_serving.example import mnist_input_data

batch_size = 100
IMAGE_WIDTH = 227
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_WIDTH * 3
CLASSES = 2


def train():
    with tf.Session() as sess:
        # tf.get_default_graph()
        images__input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
        W = tf.Variable(tf.zeros([IMAGE_PIXELS, CLASSES]))
        b = tf.Variable(tf.zeros([10]))
        y_ = tf.matmul(images__input_placeholder, W) + b
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels_placeholder, y_))
        tf.arg_max()


# ##############################################alexNet###########################################
def inference(images):  # image batchsize*227*227*3



    # Layer 1
    with tf.name_scope('layer1'):
        # conv1        #11*11*3*96 stride=4
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], 'SAME', name='conv')
        bias = tf.Variable(tf.zeros([64], dtype=tf.float32))
        biased = tf.nn.bias_add(conv, bias)
        relu_o = tf.nn.relu(biased, name='relu')

        print_activations(relu_o)
        # MAX_POOL
        layer1_output = tf.nn.max_pool(relu_o,
                                       ksize=[1, 3, 3, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='VALID',
                                       name='max_pool')
        print_activations(layer1_output)
        # tf.initialize_variables([kernel,bias]).run()
        # print('kernel: ',kernel.eval())
        # print('kernel: ',kernel.eval())

    # Layer 2
    with tf.name_scope('layer2'):
        # conv2        #11*11*3*96 stride=4
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(layer1_output, kernel, [1, 1, 1, 1], padding='SAME', name='conv')
        bias = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                           trainable=True, name='bias')
        biased = tf.nn.bias_add(conv, bias)
        relu_o = tf.nn.relu(biased, name='relu')

        print_activations(relu_o)
        # LRN
        # (TODO) add a Local Response Normalization here

        # max pool2
        layer2_output = tf.nn.max_pool(relu_o,
                                       ksize=[1, 3, 3, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='VALID',
                                       name='pool2')
        print_activations(layer2_output)

    # layer 3
    with tf.name_scope('layer3'):
        # conv3
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(layer2_output, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                           trainable=True, name='biases')
        biased = tf.nn.bias_add(conv, bias)
        layer3_output = tf.nn.relu(biased, name='relu')

        print_activations(layer3_output)

    # layer4
    with tf.name_scope('layer4'):
        # conv4
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(layer3_output, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                           trainable=True, name='biases')
        biased = tf.nn.bias_add(conv, bias)
        layer4_output = tf.nn.relu(biased, name='relu')

        print_activations(layer4_output)

    # layer5
    with tf.name_scope('layer5'):
        # conv 5
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(layer4_output, kernel, [1, 1, 1, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                           trainable=True, name='biases')
        biased = tf.nn.bias_add(conv, bias)
        layer4_output = tf.nn.relu(biased, name='relu')

        print_activations(layer4_output)

        # max pool5
        layer5_output = tf.nn.max_pool(layer4_output,
                                       ksize=[1, 3, 3, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='VALID',
                                       name='pool5')
        print_activations(layer5_output)  # (batch_size*6*6*256)


    # fc6
    with tf.name_scope('layer6'):
        fc6W = tf.Variable(tf.truncated_normal((9216, 4096), dtype=tf.float32), name='Weights')
        fc6b = tf.Variable(tf.truncated_normal([4096], dtype=tf.float32), name='Weights')
        layer6_output = tf.nn.relu_layer(tf.reshape(layer5_output, [-1, int(np.prod(layer5_output.get_shape()[1:]))]),
                                         fc6W, fc6b, name='relu_layer')
        print_activations(layer6_output)

    with tf.name_scope('layer7'):
        fc7W = tf.Variable(tf.truncated_normal((4096, 4096), dtype=tf.float32), name='Weights')
        fc7b = tf.Variable(tf.truncated_normal([4096], dtype=tf.float32), name='Weights')
        layer7_output = tf.nn.relu_layer(layer6_output, fc7W, fc7b, name='relu_layer')
        print_activations(layer7_output)
        # tf.initialize_all_variables().run()
        # print('layer7_o: ',layer7_output.eval())

    with tf.name_scope('layer8'):
        fc8W = tf.Variable(tf.truncated_normal((4096, CLASSES), dtype=tf.float32, mean=0.1), name='Weights')
        fc8b = tf.Variable(tf.truncated_normal([CLASSES], dtype=tf.float32), name='Weights')
        layer8_output = tf.nn.relu_layer(layer7_output, fc8W, fc8b, name='relu_layer')
        print_activations(layer8_output)
        # tf.initialize_all_variables().run()
        # print('fc8W: ',fc8W.eval())
        # print('layer8_o: ',layer8_output.eval())

    # sm_output = tf.nn.softmax(layer8_output, name='soft_max')
    return layer8_output, (layer7_output,layer6_output,layer5_output,layer4_output,layer3_output,layer2_output, layer1_output,fc8W)


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    #TODO add regularization
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return cross_entropy_mean

def train(total_loss,lr = 0.001):
    # get data
    # for loop
    # export model
    op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(total_loss)
    return op

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())
    pass

def main():
    batch_size = 100
    steps = 1
    sess = tf.Session()
    with sess.as_default():
        train_data, train_label = preprocess.load_train_set()
        train_data, train_label = preprocess.randomize(train_data, train_label)

        image_placeholder = tf.placeholder(dtype=tf.float32,shape=(batch_size,227,227,3))
        label_placeholder = tf.placeholder(dtype=tf.int8,shape=(batch_size))
        logits, l7654321 = inference(image_placeholder)
        layer7_t,layer6_t,layer5_t,layer4_t,layer3_t,layer2_t,layer1_t,fc8W_t = l7654321
        total_loss = loss(logits, label_placeholder)
        train_op = train(total_loss,lr=0.0000001)
        tf.initialize_all_variables().run()

        for i in range(steps):
            start_position = random.randrange(len(train_label)-batch_size)
            softmax = tf.nn.softmax(logits)

            _, loss_value, logits_value, softmax_value, l7_output,l6_output,l5_output,l1_output, fc8w = \
                sess.run([train_op, total_loss, logits, softmax, layer7_t, layer6_t, layer5_t, layer1_t, fc8W_t],
                                     feed_dict={image_placeholder: train_data[start_position:start_position+batch_size],
                                                label_placeholder: train_label[start_position:start_position+batch_size]})
            result = np.argmax(logits_value, axis=1)

            # print('l1_output:',l1_output)
            # print('l2_output:',l7654321[5])
            # print('l3_output:',l7654321[4])
            # print('l4_output:',l7654321[3])
            # print('l5_output:',l5_output)
            # print('l6_output:',l6_output)
            # np.savetxt('l6.txt',l6_output)
            print('l7_output:',l7_output)
            print('FC8W: ',     fc8w)
            print('logits:',logits_value)
            print('result',result)
            print('lables',train_label[start_position:start_position+batch_size])
            print('accuracy:',np.sum(result==train_label[start_position:start_position+batch_size]))
            print('total loss:', loss_value)

        # save model to file.
        # gd = tf.get_default_graph().as_graph_def()
        # saver = tf.train.Saver({"my-test-variable": fc8W_t})
        # saver.save(sess,'temp.ckpt')
        tf.train.write_graph(sess.graph_def, '', 'graph_pb', as_text=True)
        # print(sess.graph_def)
        minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["layer8/relu_layer"])

        tf.train.write_graph(minimal_graph, '.', 'minimal_graph.proto', as_text=False)
        tf.train.write_graph(minimal_graph, '.', 'minimal_graph.txt', as_text=True)

'''
############################# Save variables to file
saver = tf.train.Saver()
save_path = saver.save(sess, "modelckpt")
print("Model saved in file: %s" % save_path)
############ resotore variables from file
saver.restore(sess, "/tmp/model.ckpt")
ps: usr  inspect_checkpoint.py to analysis this file
'''

if __name__ == '__main__':
    import sys

    sys.exit(int(main() or 0))
