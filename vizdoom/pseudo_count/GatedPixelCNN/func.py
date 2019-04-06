import logging

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")
import os
import time

import numpy as np
import tensorflow as tf
from .network import Network
from .utils import preprocess_conf
from tqdm import tqdm
flags = tf.app.flags
from skimage.transform import resize
import argparse

# network
flags.DEFINE_integer("gated_conv_num_layers", 2, "the number of gated conv layers")
flags.DEFINE_integer("gated_conv_num_feature_maps", 16,
                     "the number of input / output feature maps in gated conv layers")
flags.DEFINE_integer("output_conv_num_feature_maps", 64, "the number of output feature maps in output conv layers")
flags.DEFINE_integer("q_levels", 8, "the number of quantization levels in the output")
# 4 used in mnist?
# training
flags.DEFINE_float("max_epoch", 100000, "maximum # of epochs")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("decay", 0.95, "decay")
flags.DEFINE_float("momentum",0.9, "momentum")
flags.DEFINE_float("epsilon", 1e-4, "epsilon")
flags.DEFINE_float("grad_norm_clip", 40.0, "grad norm of clip")

flags.DEFINE_float("grad_clip", 1, "value of gradient to be used for clipping")

# generation
# Debug
# flags.DEFINE_boolean("is_train", True, "training or testing")
flags.DEFINE_string("log_level", "INFO", "log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

# parser = argparse.ArgumentParser()
# parser.add_argument('--gated_conv_num_layers', type=int, default=2)
# parser.add_argument('--gated_conv_num_feature_maps', type=int, default=16)
# parser.add_argument('--output_conv_num_feature_maps', type=int, default=64)
# parser.add_argument('--q_levels', type=int, default=8)
# parser.add_argument('--max_epoch', type=float, default=100000)
# parser.add_argument('--learning_rate', type=float, default=1e-3)
# parser.add_argument('--decay', type=float, default=0.95)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--epsilon', type=float, default=1e-4)
# parser.add_argument('--grad_norm_clip', type=float, default=40.0)
# parser.add_argument('--grad_clip', type=float, default=1.0)
# parser.add_argument('--log_level', type=str, default='INFO')
# parser.add_argument('--random_seed', type=int, default=123)



# parser.add_argument('--entropy', action='store_true', help='whether to use entropy for exploration')
# parser.add_argument('--bz', action='store_true', help='whether to use entropy for exploration')
# parser.add_argument('--train', action='store_true', help='train the agent')
# parser.add_argument('--test', action='store_true', help='test the agent')
# parser.add_argument('--save_dir', type=str, default='./trash/')
# parser.add_argument('--env', type=str, default='simpler_basic')
# parser.add_argument('--epochs', type=int, default=20)
# parser.add_argument('--learning_steps_per_epoch', type=int, default=5000)
# parser.add_argument('--replay_memory_size', type=int, default=50000)
# parser.add_argument('--power', type=int, default=1)
# parser.add_argument('--clip', action='store_true', help='clip')
# parser.add_argument('--clip_value', type=float, default=0.99)
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--max_eps_upto', type=float, default=0.1)
# parser.add_argument('--decay_eps_upto', type=float, default=0.6)
# parser.add_argument('--exponential', action='store_true', help='exponential decay')
# parser.add_argument('--bonus', action='store_true', help='use bonus')


# conf = parser.parse_args()
conf = flags.FLAGS

# logging
logger = logging.getLogger()
logger.setLevel(conf.log_level)

# random seed
tf.set_random_seed(conf.random_seed)
np.random.seed(conf.random_seed)

# preprocess the data into 0-1
def preprocess(q_levels):
    def preprocess_fcn(images):
        # Create the target pixels from the image. Quantize the scalar pixel values into q_level indices.
        target_pixels = np.clip(((images * q_levels).astype('int64')), 0, q_levels - 1)  # [N,H,W,C]
        return (images, target_pixels)

    return preprocess_fcn

def rgb2y(image):
    assert (len(image.shape) == 3)
    assert (image.shape[-1] == 3)
    im_y = np.dot(image[..., :3], [0.229, 0.587, 0.144])
    im_y = resize(im_y, (84, 84), order=1)
    im_y = resize(im_y, (42, 42), order=1)
    im_y = im_y / 255.
    return im_y.astype(np.float32)


def collect_samples(batch_size, env, action_n, ob_shape=(42, 42)):
    samples = []
    # temporally use random policy
    for i in range(batch_size):
        action = np.random.randint(action_n)
        s, r, terminal, _ = env.step(action)
        if terminal:
            env.reset()
        s = rgb2y(s)
        samples.append(s)
        # temporally ignore reward
    samples = np.array(samples).reshape((batch_size,) + ob_shape + (1,))
    q_fun = preprocess(8)
    return q_fun(samples)                            

# I would find the value range of the image.
def process_density_images(image):
    # image = image / 255.
    # density_images = resize(image, (42, 42, 1), order=1)
    return image.reshape(-1, 42, 42, 1).astype(np.float32)
    # return image.reshape(-1, 640, 480, 1).astype(np.float32)

def process_density_input(samples):
    # NHWC thx!
    q_func = preprocess(8)
    return q_func(samples)

def get_network(dens_scope):
    preprocess_conf(conf)
    network = Network(conf, 42, 42, 1, dens_scope)
    # network = Network(conf, 640, 480, 1, dens_scope)
    return network
