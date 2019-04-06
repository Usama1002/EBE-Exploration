# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Name: Muhammad Usama
# Student Number: 20174549

import tensorflow as tf

# This class 'Estimator' defines structure and functionality for q_estimator and target_estimator neural networks.
class Estimator():
    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        with tf.variable_scope(scope):
            self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2=self._build_model()

    def _build_model(self):
        self.X_pl = tf.placeholder(shape=[None, 8, 5, 2], dtype=tf.float32, name="X")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        # X = tf.to_float(self.X_pl)
        batch_size_data = tf.shape(self.X_pl)[0]


        # The neural networks, both target as well as Q values estimator consist of 2 covolutional layers, followed by
        # a fully connected layer and then an output layer.

        # these are the parameters of out neural network.

        # First convolutional layer
        # there are 32 filters in first convolutonal layer
        W_conv1=tf.Variable(tf.truncated_normal([3, 3, 2, 32], stddev=0.1)) # filter size: 3x3, stride:1
        b_conv1=tf.Variable(tf.constant(0.1, shape=[32])) # bias

        # there are 64 filters in second convolutonal layer
        W_conv2 = tf.Variable(tf.truncated_normal([2, 2, 32, 64], stddev=0.1)) # filter size: 2x2, stride:1
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) # bias

        # Fully connected layer having 256 neurons
        W_fc1=tf.Variable(tf.truncated_normal([640, 256], stddev=0.1)) # weight matrix
        b_fc1=tf.Variable(tf.constant(0.1, shape=[256])) # bias

        # Output layer
        W_fc2 = tf.Variable(tf.truncated_normal([256, 3], stddev=0.1)) # weight matrix
        b_fc2 =tf.Variable(tf.constant(0.1, shape=[3])) # bias

        # Output of first convolutional layer with RELU activation function.
        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.X_pl, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)

        # Output of second convolutional layer with RELU activation function.
        h_conv2=tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2)

        # Flattening of output of 2nd convolutional layer
        h_conv2_flat = tf.reshape(h_conv2, [-1, 640])

        # Output of the fully connected layer
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        # Q Value: Output of the output layer
        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2

        # Here, out of all QValues, we find only those whose actions are given in the actions.pl placeholder.
        gather_indices = tf.range(batch_size_data) * tf.shape(QValue)[1] + self.actions_pl

        # passing [-1] to flatten the predictions
        self.specific_action_Q_values = tf.gather(tf.reshape(QValue, [-1]), gather_indices)

        # Loss function as described in the algorithm
        self.losses = tf.squared_difference(self.y_pl, self.specific_action_Q_values)
        self.loss = tf.reduce_mean(self.losses)

        # We are using the RMSProp optimizer as used in the original paper.
        # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.95, 1e-6) # 0.00025
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_op = self.optimizer.minimize(self.loss)

        # Returning the values including the network parameters for assigning operation.
        return tf.to_float(self.X_pl), QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2

    # This function is used to predict the QValues given the state 's'
    def predict(self, sess, s):
        return sess.run(self.QValue, feed_dict={ self.X_pl: s })

    # This function runs the optimization given input placeholder data.
    def update(self, sess, s, a, y):
        self.train_op.run(feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a })

    # This function takes another object of this class and copies its parameters to 'self' scope object using
    # tf.assign command.
    def copy_parameters(self,q_est):
        my_copier=[tf.assign(self.W_conv1 , q_est.W_conv1)  ,  tf.assign(self.b_conv1 , q_est.b_conv1) ,
                   tf.assign(self.W_conv2, q_est.W_conv2), tf.assign(self.b_conv2, q_est.b_conv2),
                   tf.assign(self.W_fc1, q_est.W_fc1), tf.assign(self.b_fc1, q_est.b_fc1),
                    tf.assign(self.W_fc2,q_est.W_fc2), tf.assign(self.b_fc2,q_est.b_fc2)]
        return my_copier