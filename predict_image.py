import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread('wxb_pic/pic_test/n/480.jpg')
pp = cv2.resize(img, (227, 227))
pp = np.asarray(pp, dtype=np.float32)
pp /= 255
pp = pp.reshape((pp.shape[0], pp.shape[1], 3))

ckpt = tf.train.get_checkpoint_state("save")
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

pred = tf.get_collection("pred")[0]
x = tf.get_collection("x")[0]
keep_prob = tf.get_collection("keep_prob")[0]

with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
      
    # Calculate accuracy for 256 mnist test images
    predict = sess.run(pred, feed_dict={x: [pp], keep_prob: 1.})
    print "Testing predict:", np.argmax(predict, 1)[0]
