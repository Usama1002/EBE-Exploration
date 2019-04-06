# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Information Theory & Machine Learning Lab, School of EE, KAIST
# written by Sae-Young Chung, 11/19/2017

import numpy as np

# breakout environment
class breakout_environment: 
    def __init__(self, nx = 5, ny = 8, nb = 3, nt = 1, nf = 2):
        self.nx = nx   # number of pixels of screen = ny * nx
        self.ny = ny
        self.nb = nb   # number of rows of bricks
        self.nt = nt   # gap at the top
        self.nf = nf   # number of most recent frames at the input layer 
        self.na = 3    # number of actions is 3: left, stay, right
        _ = self.reset()

    def reset(self):   # reset the game
        self.s = np.zeros([self.ny, self.nx, self.nf])    # two consecutive frames, the state consists of current and last frame.
        self.b = np.ones([self.nb+1, self.nx+1], dtype=np.int)  # initially all bricks are present, add dummy row and column
        self.b[-1] = 0; self.b[:,-1] = 0 # a[-1] means last element of array a. a[-2] means second last element.
        self.s[self.ny-self.nt-self.nb:self.ny-self.nt,:,:] = 1.    # draw bricks 
        self.bx, self.by = 0, self.ny - self.nt - self.nb - 1        # initial position of ball, bx=0, by=total rows - empty rows - rows having bricks, so initialliy ball is extreme left and just below the last row of bricks
        self.s[self.by, self.bx, :] = 1.                    # draw ball, mark the position of ball
        self.vx = 1    # speed of ball in x axis, -1: left, 1: right
        self.vy = -1   # speed of ball in y axis, -1: down, 1: up  , so initially ball is moving diagonally left and downwards
        self.p = self.nx // 2       # position of paddle , just in the middle of the x.axis
        self.s[0, self.p, :] = 1.    # draw paddle, set its pixel to 1 in the state space
        return np.copy(self.s)  # who not just return 'self.s' ?

    def run(self, action):      # take action (-1: left, 0: stay, 1: right), change state, return updated state, reward, terminal, and other variables
        reward = 0.
        terminal = 0
        rx, ry = 0, 0
        p0 = self.p  # paddel position before the action is taken.
        self.p = min(max(p0 + action, 0), self.nx - 1)      # update paddle position, 'max' saves from getting values lower than left x-axis limit and 'min' does that for x-axis right limit.
        self.s[:,:,1:] = self.s[:,:,:self.nf - 1]  # as the state has 'nf' frames, so when the environment is run, initial 'nf-1' frames are pushed towards right and position 0 in set vacant for new frame (frame after action), the earliest frame is thus discarded.
        self.s[0, p0, 0] = 0.   # initially set next frame to zero
        self.s[0, self.p, 0] = 1.   # update the paddle position in the current screen (screen after action)
        
        bx0, by0, vx0, vy0 = self.bx, self.by, self.vx, self.vy   # initial position and velocity of the ball
        self.bx, self.by = bx0 + vx0, by0 + vy0   # next position and velocity of the ball
        yofs = self.ny - self.nt - self.nb  # empty rows below the last bricks row

        if self.bx < 0: # # in case if ball reaches the minimum right x-axis limit, 0, invert the x_axis velocity
            self.bx = 0
            self.vx = -self.vx
        elif self.bx >= self.nx:  # in case if ball reaches the maximum right x-axis limit, invert the x_axis velocity
            self.bx -= 1
            self.vx = -self.vx
        if self.by >= self.ny:      # hitting the ceiling, invert the y_axis velocity
            self.by -= 1
            self.vy = -self.vy
            if self.nt == 0:   # in this case, the top row may have some bricks
                if self.b[-1, self.bx]: # if a brick is present where the ball is currently present
                    rx, ry = self.bx, self.nb-1
                    self.b[-1, self.bx] = 0 # now brick is not present here as the ball has hit it.
                    self.s[-1, self.bx, 0] = 0. # update position of the ball in current screen, since there is a brick, so ball can't actually move there and will stay at its previous position.
                    self.bx = bx0
                    self.vx = -vx0 # also invert the x_axis velocity.
                    reward = 1. # we also got the reward
        elif self.by == 0:     # hitting the bottom, check if game over
            cx = bx0 + (vx0 > 0)
            if self.p == cx:   # hitting the ball with the left edge of the paddle
                self.bx = bx0 - (vx0 < 0)
                self.vx = -1
                self.by = by0
                self.vy = 1
                if self.bx < 0:
                    self.bx = 0
                    self.vx = 1
            elif self.p + 1 == cx:    # hitting the ball with the right edge of the paddle
                self.bx = bx0 + (vx0 > 0)
                self.vx = 1
                self.by = by0
                self.vy = 1
                if self.bx >= self.nx:
                    self.bx = self.nx - 1
                    self.vx = -1
            else:
                terminal = 1    # game over
        elif self.by >= yofs and self.by < yofs + self.nb:
            if self.b[self.by - yofs, bx0] and self.b[by0 - yofs, self.bx]:   # if a left/right brick is to be broken
                rx, ry = self.bx, by0 - yofs
                self.b[by0 - yofs, self.bx] = 0
                self.s[by0, self.bx, 0] = 0.
                self.bx, self.by, self.vx, self.vy = bx0, by0, -vx0, -vy0
                reward = 1.
            elif self.b[self.by - yofs, self.bx]:    # if a brick in the ball's trajectory is to be broken
                rx, ry = self.bx, self.by - yofs
                self.b[self.by - yofs, self.bx] = 0
                self.s[self.by, self.bx, 0] = 0.
                if vx0 < 0 and bx0 == 0 or vx0 > 0 and bx0 == self.nx - 1:
                    self.bx, self.by, self.vx, self.vy = bx0, by0, -vx0, -vy0
                else:
                    ix = not self.b[self.by - yofs, bx0] or self.b[by0 - yofs, self.bx]
                    iy = self.b[self.by - yofs, bx0] or not self.b[by0 - yofs, self.bx]
                    if ix:
                        self.bx, self.vx = bx0, -vx0
                    if iy:
                        self.by, self.vy = by0, -vy0
                reward = 1.

        # update ball position in the current screen
        self.s[by0, bx0, 0] = 0.    # Set initial ball position to 0
        self.s[self.by, self.bx, 0] = 1.       # set new ball position to 1

        if np.sum(self.b) == 0:     # all bricks broken
            terminal = 1
        # return: new state, reward, if terminal or not, last padddle position, new paddle position, last position and velocity of ball,
        return np.copy(self.s), reward, terminal, p0, self.p, bx0, by0, vx0, vy0, rx, ry

    # interpolated ball position for animation, where t is in [0,1]
    def get_ball_pos(self, t):
        x0, y0 = self.bx + (self.vx < 0), self.by + (self.vy < 0)
        x1, y1 = x0 + self.vx, y0 + self.vy
        return t * x1 + (1-t) * x0, t * y1 + (1-t) * y0

