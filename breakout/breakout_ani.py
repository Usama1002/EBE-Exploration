# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Name: Muhammad Usama
# Student Number: 20174549

import tensorflow as tf
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from neural_network import *
from breakout_env import *
from wait import *
import argparse
parser = argparse.ArgumentParser(description='Entropy Based Exploration')
q_values_fig = plt.figure(1)

augmented_entropy = []

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    # return ( x / np.sum(x, axis=0) )

def calculate_entropy(action_values):
    print('****************** here *******************8')
    base = np.shape(action_values)[0]
    action_probabilities = softmax(action_values)
    entropy = sc.entropy(action_probabilities, base = base)
    return np.power(entropy,1)


class breakout_animation(animation.TimedAnimation):
    def __init__(self, env, max_steps,q_estimator, frames_per_step=5):

        self.number_of_steps=0

        self.env = env
        self.max_steps = max_steps

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        self.objs = []

        # boundary
        w = 0.1
        ax.plot([-w, -w, env.nx + w, env.nx + w], [0, env.ny + w, env.ny + w, 0], 'k-', linewidth=5)

        # bricks
        wb = 0.05
        self.bricks = []
        self.brick_colors = [['red'], ['blue', 'red'], ['blue', 'green', 'red'], ['blue', 'green', 'yellow', 'red'],
                             ['blue', 'green', 'yellow', 'orange', 'red'], \
                             ['purple', 'blue', 'green', 'yellow', 'brown', 'orange', 'red'],
                             ['purple', 'blue', 'green', 'yellow', 'brown', 'orange',
                              'red']]  # add more colors if needed
        for y in range(self.env.nb):
            b = []
            yp = y + (self.env.ny - self.env.nt - self.env.nb)
            for x in range(self.env.nx):
                b.append(patches.Rectangle((x + wb, yp + wb), 1 - 2 * wb, 1 - 2 * wb, edgecolor='none',
                                           facecolor=self.brick_colors[self.env.nb - 1][y]))
                ax.add_patch(b[x])
                self.objs.append(b[x])
            self.bricks.append(b)

        # ball
        self.ball = patches.Circle(env.get_ball_pos(0.), radius=0.15, color='red')
        ax.add_patch(self.ball)
        self.objs.append(self.ball)

        # score text
        self.text = ax.text(0.5 * env.nx, 0, '', ha='center')
        self.objs.append(self.text)

        # game over text
        self.gameover_text = ax.text(0.5 * env.nx, 0.5 * env.ny, '', ha='center')
        self.objs.append(self.gameover_text)

        self.frames_per_step = frames_per_step
        self.total_frames = self.frames_per_step * self.max_steps

        # paddle
        self.paddle = patches.Rectangle((env.p, 0.5), 1, 0.5, edgecolor='none', facecolor='red')
        ax.add_patch(self.paddle)

        # for early termination of animation
        self.iter_objs = []
        self.iter_obj_cnt = 0

        # interval = 50msec
        animation.TimedAnimation.__init__(self, fig, interval=50, repeat=False, blit=False)

    def _draw_frame(self, k):
        if self.terminal:
            return
        if k == 0:
            self.iter_obj_cnt -= 1
        if k % self.frames_per_step == 0:

            # this is the modification mae in this code, now instead of acting randomly, the agent acts greedily
            # basd on the policy given by q_estimator predictions.
            # we give our state to the q_estimator, and it returns us the q_values for all actions.
            q_values = q_estimator.predict(sess, np.expand_dims(self.env.s, axis=0))
            # getting the greedy action out of q-values
            self.a = np.random.choice(np.where(q_values[0] == np.max(q_values))[0]) - 1
            self.number_of_steps+=1
            self.p = self.env.p
            self.pn = min(max(self.p + self.a, 0), self.env.nx - 1)

            plt.figure(1)
            # actions = [1,2,3]
            actions = ['move left', 'stay where you are', 'move right']
            print(self.a)
            # max_actions = np.where(q_values[0] == np.max(q_values))[0]
            colors = ['g' if -1 == self.a else '0.75',
                      'g' if 0 == self.a else '0.75',
                      'g' if 1 == self.a else '0.75']
            plt.gca().clear()
            plt.bar(actions, q_values[0], color=colors, width=0.3)
            plt.ylim(top=6)
            plt.xlabel('Actions to Take')
            plt.ylabel('Q-Values')
            plt.title('Q-Values for Available Actions in a State')
            # plt.figure(2)
            # plt.plot(augmented_entropy.append(calculate_entropy(q_values[0])))
            plt.show(block=False)
            


        t = (k % self.frames_per_step) * 1. / self.frames_per_step
        self.ball.center = self.env.get_ball_pos(t)
        self.paddle.set_x(t * self.pn + (1 - t) * self.p)

        if k % self.frames_per_step == self.frames_per_step - 1:
            sn, reward, terminal, p0, p, bx0, by0, vx0, vy0, rx, ry = self.env.run(self.a)
            self.sum_reward += reward
            if reward > 0.:
                self.bricks[ry][rx].set_facecolor('none')
                self.text.set_text('Score: %d' % self.sum_reward)
            if terminal:
                self.terminal = terminal
                self.gameover_text.set_text('Game Over')
                for _ in range(self.total_frames - k - 1):
                    next(self.iter_objs[self.iter_obj_cnt])  # for early termination of animation (latest iterator is used first)

        self._drawn_artists = self.objs

    def new_frame_seq(self):
        iter_obj = iter(range(self.total_frames))
        self.iter_objs.append(iter_obj)
        self.iter_obj_cnt += 1
        return iter_obj

    def _init_draw(self):
        sn1 = self.env.reset()
        q_values = q_estimator.predict(sess, np.expand_dims(sn1, axis=0))
        self.sum_reward = 0.
        self.p = self.env.p  # current paddle position
        self.pn = self.p  # next paddle position
        self.a = np.random.choice(np.where(q_values[0] == np.max(q_values))[0]) - 1  # action
        self.terminal = 0

        for y in range(self.env.nb):
            for x in range(self.env.nx):
                self.bricks[y][x].set_facecolor(self.brick_colors[self.env.nb - 1][y])

        self.ball.center = self.env.get_ball_pos(0.)
        self.paddle.set_x(self.p)

        self.text.set_text('Score: 0')
        self.gameover_text.set_text('')

# _______________________________ MAIN ____________________________________________________

# managing command line arguments

parser.add_argument('--save_path', type=str, default='./data/unspecified_path/')
parser.add_argument('--episodes', type=int, default=2500)
parser.add_argument('--best', action='store_true')
args = parser.parse_args()



restore_model_percentages = [20,40,60,80,100]
save_path = args.save_path
num_episodes = args.episodes

sess = tf.InteractiveSession()
q_estimator=Estimator()
saver=tf.train.Saver()

# if args.best:
#     saver.restore(sess, save_path=save_path + 'best_model.ckpt')
#     env = breakout_environment(5, 8, 3, 1, 2)
#     ani = breakout_animation(env, 200, q_estimator)
#     ani.save(save_path + 'video_best' + '.mp4', dpi=200)
#     # plt.show(block=False)
#     print('Number of Steps Taken: %.f' % ani.number_of_steps)
#     print('******************* Generated Video for BEST **********************')
# else:
#     for percentage in restore_model_percentages:
#         i_episode = int(percentage * 0.01 * num_episodes)
#         saver.restore(sess, save_path=save_path+'model.ckpt-'+str(i_episode))
#         env = breakout_environment(5, 8, 3, 1, 2)
#         ani = breakout_animation(env, 200, q_estimator)
#         ani.save(save_path + 'video_steps_' + str(i_episode) + '.mp4', dpi=200)
#         # plt.show(block=False)
#         print('Number of Steps Taken: %.f' % ani.number_of_steps)
#         print('******************* Generated Video for Steps: '+ str(i_episode) + '*******************')

saver.restore(sess, save_path=save_path + 'best_model.ckpt')
env = breakout_environment(5, 8, 3, 1, 2)
ani = breakout_animation(env, 200, q_estimator)
# ani.save(save_path + 'video_best' + '.mp4', dpi=200)

plt.show(block=False)
# animation.FuncAnimation(ani._draw_frame, 100)
print('Number of Steps Taken: %.f' % ani.number_of_steps)
print('******************* Generated Video for BEST **********************')

wait('Press enter to quit')
