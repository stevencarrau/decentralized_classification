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
        self.highlight_reel = HighlightReel(num_items=5)

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

class HighlightReel:
    """
    Stores an array of data about the top `num_items` timesteps
    where the change in beliefs was the highest over the episode.
    """

    # dictates what column indices map to in the 2D reel np array
    ITEM_LABELS_TO_IDX = {"time_step": 0, "max_delta": 1, "prev_state": 2, "next_state": 3, "trigger": 4}
    NUM_ITEM_LABELS = len(ITEM_LABELS_TO_IDX)
    EMPTY_ITEM = np.full((NUM_ITEM_LABELS), -1, dtype=float)

    def __init__(self, num_items=10):
        self.reel_length = num_items
        self.reel = np.full((self.reel_length, self.NUM_ITEM_LABELS), self.EMPTY_ITEM, dtype=float)

    def __str__(self):
        return self.reel.__str__()

    def sort(self):
        """
        sort self.reel based off of criteria.
        currently, reel is sorted on the max_delta column
        """
        column_idx = self.ITEM_LABELS_TO_IDX["max_delta"]
        self.reel = self.reel[self.reel[:, column_idx].argsort()]

    def add_item(self, **item_args):
        """
        Given some arguments, generates a new np array to insert into self.reel.
        item_args should be a dict of form {"time_step": 15, "max_delta": 0.23, ...} following
        the labels in ITEM_LABELS_TO_IDX
        """
        assert item_args.keys() == self.ITEM_LABELS_TO_IDX.keys(), "Labels must exactly match to those in " \
                                                              "ITEM_LABELS_TO_IDX"

        # prepare 1D numpy array from dict in parameters using ITEM_LABELS_TO_IDX as an indexing guide
        reel_item = self.EMPTY_ITEM.copy()
        for label in item_args:
            label_idx = self.ITEM_LABELS_TO_IDX[label]
            value_of_label = item_args[label]
            reel_item[label_idx] = value_of_label

        empty_idxs = np.where(self.reel == self.EMPTY_ITEM)[0]
        # if there are still empty spots, fill the first one since nothing need be replaced
        if len(empty_idxs) != 0:
            empty_idx = empty_idxs[0]
            self.reel[empty_idx] = reel_item
        else:
            # otherwise, replace item 0 as it will always be the item in
            # the list with the smallest max delta due to sorting
            self.reel[0] = reel_item

        # sort always as a new item populates -- shouldn't be too time inefficient since
        # only `num_item` subarrays which should be <= 10 (?)
        self.sort()
