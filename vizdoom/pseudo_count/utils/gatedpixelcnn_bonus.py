import cv2
import numpy as np
from math import pow, exp
from GatedPixelCNN.func import process_density_images, process_density_input, get_network
from GatedPixelCNN.utils import save_images
import logging
import torch
import tensorflow as tf
from tensorboardX import SummaryWriter
from skimage.transform import resize

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


class PixelBonus(object):
    """
    Computes and updates PixelCNN++ bonus - taken from https://github.com/pclucas14/pixel-cnn-pp
    """
    def __init__(self, FLAGS, sess, num_actions):
        # init model
        self.sess = sess
        self.density_model = get_network("density")
        self.flags = FLAGS
        self.writer = SummaryWriter()        
        self.frame_shape = (FLAGS.img_height, FLAGS.img_width)

    def bonus(self, obs, action, t, num_actions):
        step = t
        # print(obs.shape)
        frame = resize(obs, (self.flags.img_height, self.flags.img_width), order=1)
        last_frame = process_density_images(frame)
        density_input = process_density_input(last_frame)

        prob = self.density_model.prob_evaluate(self.sess, density_input, True)
        prob_dot = self.density_model.prob_evaluate(self.sess, density_input)
        prob += 1e-8
        prob_dot += 1e-8
        pred_gain = np.sum(np.log(prob_dot) - np.log(prob))
        self.writer.add_scalar('data/loss', -np.sum(np.log(prob)), t)
        self.writer.add_scalar('data/PG', pred_gain, t)
        psc_reward = pow((exp(0.1*pow(step + 1, -0.5) * max(0, pred_gain)) - 1), 0.5)
        return psc_reward


    def sample_images(self, n, t):
        data = np.zeros([n * n, self.flags.img_height, self.flags.img_width, 1])
        sample = self.density_model.generate_samples(self.sess, data)
        print(sample)
        save_images(sample, self.flags.img_height, self.flags.img_width, n, n, t=t)


    # def sample_images(self, n):
    #     data = 
        
    #     data = torch.zeros(n, 1, self.flags.img_height, self.flags.img_width)
    #     data = data.cuda()
    #     sample_op = lambda x: sample_from_discretized_mix_logistic_1d(x, self.flags.nr_logistic_mix)
    #     for i in range(self.flags.img_height):
    #         for j in range(self.flags.img_width):
    #             data_v = Variable(data, volatile=True)
    #             out = self.model(data_v, sample=True)
    #             out_sample = sample_op(out)
    #             data[:, :, i, j] = out_sample.data[:, :, i, j]
    #     rescaling_inv = lambda x: torch.clamp((x * 8.).int().float(), 0., 7.) / 8.
    #     print(data)
    #     #print(rescaling_inv(data).cpu().numpy())
    #     return rescaling_inv(data)
                             
    

