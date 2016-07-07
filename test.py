from importData import Dataset
testing = Dataset('wxb_pic/pic_test', '.jpg')

import tensorflow as tf
import numpy as np

# Parameters
batch_size = 1

ckpt = tf.train.get_checkpoint_state("save")
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

pred = tf.get_collection("pred")[0]
x = tf.get_collection("x")[0]
keep_prob = tf.get_collection("keep_prob")[0]

# Launch the graph
# with tf.Session() as sess:
sess = tf.Session()
saver.restore(sess, ckpt.model_checkpoint_path)

# test
step_test = 1
while step_test * batch_size < len(testing):
    testing_ys, testing_xs = testing.nextBatch(batch_size)
    predict = sess.run(pred, feed_dict={x: testing_xs, keep_prob: 1.})
    print "Testing label:", testing.label2category[np.argmax(testing_ys, 1)[0]]
    print "Testing predict:", testing.label2category[np.argmax(predict, 1)[0]]
    step_test += 1
