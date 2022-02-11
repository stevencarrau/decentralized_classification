import numpy as np
from math import floor,log
from math import pi
from util import Util

def entropy(dist_in):
    H = 0
    for i in dist_in:
        if i==0:
            h = 0
        else:
            h = i*log(i,len(dist_in))
        H -= h
    return H

class Agent:
    def __init__(self, c_set, label, char_name, bad_i, mdp, state, t_i, states, state_keys,mc_dict, agent_idx=0):
        self.label = label
        self.char_name = char_name
        self.c_set = c_set
        # self.c_i = self.c_set[0]
        self.b_i = bad_i
        self.t_i = t_i
        self.belief_values = np.ones((len(mc_dict[0]), 1)) / len(mc_dict[0])
        self.belief_ent = entropy(self.belief_values)
        self.belief = 0  # All agents presumed innocent to begin with
        self.max_delta = 0
        self.agent_idx = agent_idx
        self.state = state
        self.mdp = mdp
        self.mc_dict = mc_dict
        self.track_queue = []
        self.states = states
        self.state_keys = state_keys
        self.dis = Util.prod2dis(state,states)
        self.alpha = 1.0

    def likelihood(self, a, next_s, mc_dict):
        return np.array([m_i[(self.state, next_s)] for m_i in mc_dict[a]]).reshape((-1, 1))

    def update_value(self, a, next_s):
        belief = self.belief_values
        likelihood = self.likelihood(a, next_s, self.mc_dict)
        new_belief = (1-self.alpha)*belief + self.alpha*np.multiply(belief, likelihood)
        new_belief = new_belief / np.sum(new_belief)
        self.max_delta = np.max(np.abs(new_belief - self.belief_values))
        self.belief_values = new_belief
        self.belief_ent = entropy(self.belief_values)


    ## Belief update rule for each agent
    def update_belief(self, belief, bad_idx):
        self.belief = belief[bad_idx][0]
        if self.belief_line:
            val = [75 * b_i[0] + 25 for b_i in belief]
            val += val[:1]
            angles = [n / float(len(belief)) * 2 * pi for n in range(len(belief))]
            angles += angles[:1]
            self.belief_line.set_data(angles, val)
            self.belief_line.set_zorder(3)
            self.belief_fill.set_xy(np.array([angles, val]).T)
            self.belief_line.set_zorder(2)
            [b_a.set_visible(False) for b_a in self.belief_artist_set]
            self.belief_artist_set[self.dis].set_visible(True)
            self.belief_artist_set[self.dis].set_zorder(10)
            f = lambda i: belief[i]
            if max(belief) > 0.75:
                if max(range(len(belief)), key=f) == self.agent_idx:
                    self.belief_text.set_color('green')
                else:
                    self.belief_text.set_color('red')
                if self.belief > 0.75:
                    self.belief_line.set_color('red')
                    self.belief_fill.set_color('red')
                else:
                    self.belief_line.set_color('green')
                    self.belief_fill.set_color('green')
            else:
                self.belief_line.set_color('yellow')
                self.belief_fill.set_color('yellow')
                self.belief_text.set_color('black')
            return [self.belief_line, self.belief_fill,self.belief_text] + self.belief_artist_set
        return None

    def init_belief_plt(self, l_i, l_f, l_a, l_t):
        self.belief_line = l_i
        self.belief_line.set_visible(False)
        self.belief_fill = l_f
        self.belief_fill.set_visible(False)
        self.belief_artist_set = l_a
        [b_a.set_visible(False) for b_a in self.belief_artist_set]
        self.belief_text = l_t
        self.belief_text.set_visible(False)

    def activate_belief_plt(self):
        self.belief_line.axes.set_visible(True)
        self.belief_line.set_visible(True)
        self.belief_fill.set_visible(True)
        self.belief_artist_set[self.dis].set_visible(True)
        self.belief_text.set_visible(True)

    def deactivate_belief_plt(self):
        self.belief_line.axes.set_visible(False)
        self.belief_line.set_visible(False)
        self.belief_fill.set_visible(False)
        self.belief_artist.set_visible(False)
        self.belief_text.set_visible(False)