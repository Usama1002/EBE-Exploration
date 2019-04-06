import cv2
from math import pow, exp
from pixel_cnn_pp.model import PixelCNN
from pixel_cnn_pp.utils import *
import logging
from torch.autograd import Variable
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

class DotDict(object):
    def __init__(self, dict):
        self.dict = dict

    def __getattr__(self, name):
        return self.dict[name]

    def update(self, name, val):
        self.dict[name] = val

    # can delete this later
    def get(self, name):
        return self.dict[name]


# specify command line arguments using flags
FLAGS = DotDict({
    'img_height': 42,
    'img_width': 42,
    'channel': 1,
    'data': 'mnist',
    'conditional': False,
    'num_classes': None,
    'filter_size': 3,
    'init_fs': 7,
    'f_map': 16,
    'f_map_fc': 16,
    'colors': 8,
    'parallel_workers': 1,
    'layers': 3,
    'epochs': 25,
    'batch_size': 16,
    'model': '',
    'data_path': 'data',
    'ckpt_path': 'ckpts',
    'samples_path': 'samples',
    'summary_path': 'logs',
    'restore': True,
    'nr_resnet': 1,
    'nr_filters': 16,
    'nr_logistic_mix': 5,
    'resnet_nonlinearity': 'concat_elu',
    'lr_decay': 0.999995,
    'lr': 0.00002,
    'num_ds': 1,
})

class PixelBonus(object):
    """
    Computes and updates PixelCNN++ bonus - taken from https://github.com/pclucas14/pixel-cnn-pp
    """
    def __init__(self, FLAGS, num_actions):
        # init model
        # self.X = tf.placeholder(
        #     tf.float32,
        #     shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.channel])
        self.model = PixelCNN(nr_resnet=FLAGS.nr_resnet, nr_filters=FLAGS.nr_filters,
                              nr_logistic_mix=FLAGS.nr_logistic_mix,
                              resnet_nonlinearity=FLAGS.resnet_nonlinearity,
                              input_channels=FLAGS.channel, num_ds=FLAGS.num_ds,
                              num_actions=num_actions)
        self.model = self.model.cuda()
        self.flags = FLAGS
        self.writer = SummaryWriter()
        
        # init optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=FLAGS.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=FLAGS.lr_decay)

        # loss op
        self.loss_op = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
        ############# OLD STUFF #######################################################
        # self.optimizer = tf.train.RMSPropOptimizer(
        #     learning_rate=1e-3,decay=0.95,momentum=0.9).minimize(self.model.loss)

        # make sure GPU doesn't use all of the available memory
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # self.sess.run(tf.global_variables_initializer())
        self.frame_shape = (FLAGS.img_height, FLAGS.img_width)
        # self.max_val = np.finfo(np.float32).max - 1e-10

    def bonus(self, obs, a, t, num_actions):
        """
        Calculate exploration bonus with observed frame
        :param obs:
        :return:
        """
        # reshape image so that it's 42 x 42
        frame = cv2.resize(obs, self.frame_shape)

        # [0,1] pixel values
        frame = (np.reshape(frame, [1, FLAGS.img_height, FLAGS.img_width]) / 255.)
        frame = np.expand_dims(frame, 0)  # (N, Y, X, C
        # frame = np.array(frame)
        # print(frame)
        # print(frame.shape)
        frame = torch.from_numpy(frame)
        frame = frame.type(torch.FloatTensor)
        frame = frame.cuda()
        frame = Variable(frame)

        action = np.zeros((1, num_actions))
        action[0, a] = 1
        action = torch.from_numpy(action)
        action = action.type(torch.FloatTensor)
        action = action.cuda()

        # compute PG
        # log_prob = self.model(frame)
        log_prob = self.density_model_logprobs(frame, action, t, update=True)

        # train a single additional step with the same observation; no update
        # log_recoding_prob = self.model(frame, update=False)
        log_recoding_prob = self.density_model_logprobs(frame, action, t, update=False)

        # compute prediction gain
        pred_gain = max(0, log_recoding_prob - log_prob)
        # print('PG', (log_recoding_prob - log_prob).data[0])
        self.writer.add_scalar('data/PG', log_recoding_prob - log_prob, t)

        # print('pred_gain', pred_gain)

        # save log loss
        # nll = self.sess.run(self.model.nll, feed_dict={self.X: frame})

        # calculate intrinsic reward
        intrinsic_reward = pow((exp(0.1*pow(t + 1, -0.5) * pred_gain) - 1), 0.5)

        # print('intrinsic_reward', intrinsic_reward)

        return intrinsic_reward

    def density_model_logprobs(self, img, a, t, update=False):
        """
        compute log loss WITHOUT updating parameters
        :param img:
        :return:
        """
        if update:
            #self.model.train(True)
            self.model.eval()
            # torch.cuda.synchronize() # TODO: is this necessary??
            output = self.model(img, a)
            loss = self.loss_op(img, output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # print 'loss', loss.data[0]
            self.writer.add_scalar('data/loss', loss, t)
            # logprob = loss
            # _, logprob, target_idx = self.sess.run([
            #     self.optimizer, self.model.log_probs, self.model.target_idx], feed_dict={
            #     self.X: img})
        else:
            #self.model.train(True)
            self.model.eval()
            output = self.model(img, a)
            loss = self.loss_op(img, output)
            # logprob = loss
            # logprob, target_idx = self.sess.run([
            #     self.model.log_probs, self.model.target_idx], feed_dict={self.X: img})

        # print(logprob)
        # print loss.data[0]
        return loss.data[0]
        # return logprob.data[0]
        # pred_prob = logprob[np.arange(FLAGS.img_height * FLAGS.img_width),
        #                     target_idx].sum()

        # return pred_prob
