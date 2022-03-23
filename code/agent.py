import numpy as np
from math import floor,log
from math import pi
from util import Util
from darpa_model import track_outs

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

        # fields for the network graph
        # cumulative number of times that other agents stepped in a neighboring state at that time step, but just
        # worry about the state and the frequency of being in that other state, not about the agents themselves
        self.neighboring_states_count = {}

        # cumulative number of times that other agents stepped in a neighboring state at that time step, but don't
        # worry about the states: just about the frequency of being in *a* neighboring state, and the agents
        self.agents_in_neighboring_states_count = {}

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

    def livesim_step_update(self, simulation, agent_idx, all_agent_locations):
        """
        Called during grid_update() in live_sim.py to update agent
        beliefs, belief plots, and populate track queues (if required).
        simulation: Should be a Simulation object, should probably be
                    SimulationRunner.instance.
        agent_idx: The idx of the agent inside the agents list in live_sim.
                    May not always be equivalent to self.agent_idx (agent_idx is
                    the location in the agents array but agent.agent_idx is the actual
                    id tag of the agent)
        all_agent_locations: A dict from agent.agent_idx -> prod2state(agent.state).
                            Basically a mapping of all agents' previous locations before
                            any updates.
        Returns an array, write_objects.
        """
        write_objects = []
        ncols = simulation.ncols
        building_squares = simulation.building_squares

        if len(self.track_queue) == 0:
            # if we're running in highlight mode, the pre-loaded track queue
            # has been finished (the episode is done playing), so stop the animation
            if self.highlight_mode:
                # increment time step to signal that we are done with the highlight episode
                # [scroll up a bit]
                # simulation.pause_flag = True
                simulation.time_step += 1
                simulation.counter_text.set_text('{}'.format(simulation.time_step))
                # update the belief plot to the new beliefs to show the change in beliefs
                # that happened as an effect of the current highlight episode (the new plot
                # will have the most recent loaded beliefs after the episode)
                new_beliefs = self.highlight_prev_beliefs + self.highlight_delta_beliefs
                self.update_belief(new_beliefs, -2)

                # don't continue with rest of code since that will be sampling from
                # mdp etc. -- we just want to load past data
                return write_objects

            print("agent idx:", self.agent_idx)
            next_s = self.mdp.sample(self.state, simulation.ani.event)
            # find the states that are `dist` away from the agents' current state
            dist = 1
            valid_neighbor_states = Util.prod2stateSet(self.mdp.neighbor_states(self.state, dist),
                                     self.states)
            print("neighboring states before:", self.neighboring_states_count)
            print("agent in neighboring states count before:", self.agents_in_neighboring_states_count)
            print("neighbor states:", valid_neighbor_states)  # States in the environment that are dist away
            print("all_agent_locations:", all_agent_locations)
            for idx in all_agent_locations:
                # we want to only consider agents that aren't `self` and are within
                # `dist` from `self`'s state
                neighbor_state = all_agent_locations[idx]
                if (idx == self.agent_idx) or (neighbor_state not in valid_neighbor_states):
                    continue
                # another agent stepped in a neighboring state: increase the freq for that neighboring state
                self.neighboring_states_count[neighbor_state] = self.neighboring_states_count.get(neighbor_state, 0) + 1
                # another agent stepped into a neighboring state: increase the freq for that agent
                self.agents_in_neighboring_states_count[idx] = self.agents_in_neighboring_states_count.get(idx, 0) + 1

            print("updated neighboring_states:", self.neighboring_states_count)
            print("updated agent in neighboring states count:", self.agents_in_neighboring_states_count)

            self.track_queue += track_outs(
                (Util.prod2state(self.state, self.states), Util.prod2state(next_s, self.states)))
            if agent_idx == 0:
                simulation.pause_flag = True
                simulation.time_step += 1
                simulation.counter_text.set_text('{}'.format(simulation.time_step))
                write_objects += [simulation.counter_text]
            if self.track_queue[0] in simulation.observable_states:
                prev_beliefs = self.belief_values
                self.update_value(simulation.ani.event, next_s)
                print('{} at {}: {}'.format(agent_idx, simulation.time_step, self.max_delta))
                write_objects += self.update_belief(self.belief_values, -2)
                new_beliefs = self.belief_values
                delta_beliefs = new_beliefs - prev_beliefs
                delta_threat_belief = delta_beliefs[4]
                self.highlight_reel.add_item(time_step=simulation.time_step, max_delta=self.max_delta,
                                              prev_state=self.state, next_state=next_s,
                                              trigger=simulation.ani.event, prev_beliefs=prev_beliefs,
                                              delta_beliefs=delta_beliefs, delta_threat_belief=delta_threat_belief)
            self.state = next_s
            self.dis = Util.prod2dis(self.state, self.states)
            print("\n")
        non_write_dis = [0, 1, 2]
        non_write_dis.pop(Util.prod2dis(self.state, self.states))
        write_objects += [self.c_set[c_j].set_visible(False) for c_j in non_write_dis]
        c_i = self.c_set[Util.prod2dis(self.state, self.states)]
        c_i.set_visible(True)
        b_i = self.b_i
        self.activate_belief_plt()
        b_i.set_visible(True)
        text_i = self.t_i
        if simulation.ani.moving:
            agent_pos = self.track_queue.pop(0)
        else:
            agent_pos = self.track_queue[0]
        loc = tuple(reversed(Util.coords(agent_pos - 30, ncols)))
        # Use below line if you're working with circles
        b_i.set_center([loc[0] + 1, loc[1] - 1])

        # Use this line if you're working with images
        c_i.xy = loc
        c_i.xyann = loc
        c_i.xybox = loc

        if agent_pos in building_squares:
            c_i.offsetbox.image.set_alpha(0.35)
        else:
            c_i.offsetbox.image.set_alpha(1.0)

        self.c_set[Util.prod2dis(self.state, self.states)] = c_i
        if self.belief > 0.75:
            b_i.set_visible(True)
        else:
            b_i.set_visible(False)
        write_objects += [c_i, b_i]

        return write_objects

    class HighlightReel:
        """
        Inner class of Agent
        Stores an array of data about the top `num_items` timesteps
        where the change in beliefs was the highest over the episode.
        """

        # dictates what column indices map to in the 2D reel np array
        ITEM_LABELS_TO_IDX = {"time_step": 0, "max_delta": 1, "prev_state": 2, "next_state": 3, "trigger": 4,
                              "prev_beliefs": 5, "delta_beliefs": 6, "delta_threat_belief": 7}
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
            column_idx = self.ITEM_LABELS_TO_IDX["delta_threat_belief"]
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