from importData import Dataset
import inference
training = Dataset('wxb_pic/yiwu', '.jpg')
testing = Dataset('wxb_pic/pic', '.jpg')

import tensorflow as tf

# Parameters
learn_rate = 0.001
decay_rate = 0.1
batch_size = 64
display_step = 20

n_classes = training.num_labels # we got mad kanji
dropout = 0.8 # Dropout, probability to keep units
imagesize = 227
img_channel = 3

x = tf.placeholder(tf.float32, [None, imagesize, imagesize, img_channel])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

pred = inference.alex_net(x, keep_prob, n_classes, imagesize, img_channel)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learn_rate, global_step, 1000, decay_rate, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
saver = tf.train.Saver()
tf.add_to_collection("x", x)
tf.add_to_collection("y", y)
tf.add_to_collection("keep_prob", keep_prob)
tf.add_to_collection("pred", pred)
tf.add_to_collection("accuracy", accuracy)

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step < 3000:
        batch_ys, batch_xs = training.nextBatch(batch_size)

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            rate = sess.run(lr)
            print "lr " + str(rate) + " Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

        if step % 1000 == 0:
            saver.save(sess, 'save/model.ckpt', global_step=step*batch_size)
        step += 1
    print "Optimization Finished!"
    step_test = 1
    while step_test * batch_size < len(testing):
        testing_ys, testing_xs = testing.nextBatch(batch_size)
        print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: testing_xs, y: testing_ys, keep_prob: 1.})
        step_test += 1
