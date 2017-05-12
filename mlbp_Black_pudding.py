"""
Mllion Live event Border Prediction - Project MLBP
written by LynAlpha twt: @LynAlpha
ver. Black pudding
least edit 17 05 11
"""
"""
Training data set was obtained from mlborder.com
Using that data was allow.
Thanks to @mlborder
"""
"""
Edit History

ver. Anago
17 05 05

ver. Black pudding
17 05 08

Uploading to Github
17 05 12
"""
import tensorflow as tf
import numpy as np
import os
import random

#setup

tf.set_random_seed(765)

xy = np.loadtxt('training_set_Black_pudding.csv', delimiter=',', dtype=np.float32)
test_xy = np.loadtxt('test_set_Black_pudding.csv', delimiter=',', dtype=np.float32)

train_x_batch = xy[:,0:-1]
train_y_batch = xy[:,-1:]

test_x = test_xy[:]

X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])
dropout_rate = tf.placeholder(tf.float32)

learning_rate = 0.00001
training_epochs = 100
batch_size = 4

CHECK_POINT_DIR = TB_SUMMARY_DIR = './netLog'

print("success to load data")

#neural net information

W1 = tf.get_variable("W1", shape=[5, 1024],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([1024]))
_L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(_L1, dropout_rate)

W2 = tf.get_variable("W2", shape=[1024, 1024],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([1024]))
_L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(_L2, dropout_rate)


W3 = tf.get_variable("W3", shape=[1024, 1024],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([1024]))
_L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(_L3, dropout_rate)


W4 = tf.get_variable("W4", shape=[1024, 1024],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([1024]))
_L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(_L4, dropout_rate)

W5 = tf.get_variable("W5", shape=[1024, 1024],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([1024]))
_L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L5 = tf.nn.dropout(_L5, dropout_rate)

W6 = tf.get_variable("W6", shape=[1024, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L5, W6) + b6

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

tf.summary.scalar("loss", cost)

last_epoch = tf.Variable(0, name='last_epoch')

# initialize
summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('success to initialize')

cost_list = list()
writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)
global_step = 0

# Saver and Restore
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECK_POINT_DIR)

if checkpoint and checkpoint.model_checkpoint_path:
    try:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    except:
        print("Error on loading old network weights")
else:
    print("Could not find old network weights")

start_from = sess.run(last_epoch)

# train
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(xy) / batch_size)
    
    for i in range(total_batch):
        feed_dict = {X: train_x_batch, Y: train_y_batch, dropout_rate: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
        writer.add_summary(s, global_step=global_step)
        global_step += 1
        
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    cost_list.append(avg_cost)
    if epoch % 10 == 9:
        print("Saving network...")
        sess.run(last_epoch.assign(epoch + 1))
        if not os.path.exists(CHECK_POINT_DIR):
            os.makedirs(CHECK_POINT_DIR)
        saver.save(sess, CHECK_POINT_DIR + "/model", global_step=i)

#save the costs
print('Learning Finished!')
np.savetxt('burned.csv', cost_list, fmt='%.8f', delimiter=',', newline='\n')

#test set and predict
predict = sess.run(hypothesis, feed_dict={X: test_x, dropout_rate : 1})
print('prediction: ', predict)
