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

        # below attributes are for showing the episodes in the highlight
        # we store them here instead of HighlightReel because self.highlight_reel
        # gets automatically updated while the sim runs. Storing them in Agent allows
        # us to set aside preloaded data, which is needed for highlighting past movements
        self.highlight_reel = self.HighlightReel(num_items=5)
        self.highlight_mode = False
        self.highlight_triggers = None
        self.highlight_time_step = None
        self.highlight_prev_beliefs = None
        self.highlight_delta_beliefs = None

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
        Inner class of Agent
        Stores an array of data about the top `num_items` timesteps
        where the change in beliefs was the highest over the episode.
        """

        # dictates what column indices map to in the 2D reel np array
        ITEM_LABELS_TO_IDX = {"time_step": 0, "max_delta": 1, "prev_state": 2, "next_state": 3, "trigger": 4,
                              "prev_beliefs": 5, "delta_beliefs": 6}
        NUM_ITEM_LABELS = len(ITEM_LABELS_TO_IDX)
        EMPTY_ITEM = np.full((NUM_ITEM_LABELS), -1, dtype=object)

        def __init__(self, num_items):
            # make sure the class finals are structured right
            subarray_idxs = [val for val in self.ITEM_LABELS_TO_IDX.values()]
            assert list(set(subarray_idxs)) == subarray_idxs, "Make sure values in ITEM_LABELS_TO_IDX are distinct, " \
                                                              "these dictate the column indices in each subarray."
            self.NUM_ITEM_LABELS = len(self.ITEM_LABELS_TO_IDX)
            self.EMPTY_ITEM = np.full((self.NUM_ITEM_LABELS), -1, dtype=object)
            # set class variables
            self.reel_length = num_items
            self.reel = np.full((self.reel_length, self.NUM_ITEM_LABELS), self.EMPTY_ITEM, dtype=object)

        def __str__(self):
            string_repr = ""
            for reel_item_idx in range(self.reel_length):
                string_repr += self.reelitem2dict(reel_item_idx).__str__() + "\n"
            return string_repr

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

            # ignore episodes with timestep 0 since they will by default
            # probably always have a high max_delta
            if item_args["time_step"] == 0:
                return

            # prepare 1D numpy array from dict in parameters using ITEM_LABELS_TO_IDX as an indexing guide
            reel_item = self.EMPTY_ITEM.copy()
            for label in item_args:
                label_idx = self.ITEM_LABELS_TO_IDX[label]
                value_of_label = item_args[label]
                reel_item[label_idx] = value_of_label

            # replace item 0: is either EMPTY_ITEM or the item in
            # the list with the smallest max delta due to sorting invariant
            self.reel[0] = reel_item

            # sort always as a new item populates -- shouldn't be too time
            # inefficient since self.reel only has `num_item` subarrays which should be <= 10 (?)
            self.sort()

        def get_item_value(self, i, label):
            """returns self.reel[i][`label`], at whatever index `label` is mapped to"""
            assert label in self.ITEM_LABELS_TO_IDX.keys(), f"label must be in {self.ITEM_LABELS_TO_IDX.keys()}"

            return self.reel[i][self.ITEM_LABELS_TO_IDX[label]]

        def reelitem2dict(self, i):
            """Returns a dictionary when trying to display subarrays of the reel"""
            dictionary = {}
            for label in self.ITEM_LABELS_TO_IDX:
                dictionary[label] = self.reel[i][self.ITEM_LABELS_TO_IDX[label]]
            return dictionary