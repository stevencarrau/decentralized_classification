import itertools
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy import stats
import os

class ProbablilityNotOne(Exception):
	pass


class Observation:
    def __init__(self, idx, measured, expected_position, bimodal_bad, bimodal_good):
        self.idx = idx
        self.measured = measured
        self.expected_position = expected_position
        self.bimodal_bad = bimodal_bad
        self.bimodal_good = bimodal_good

class Observe:
    def get_observations(agents, sensors, t):
        observe_set = []
        for agent_idx, agent in enumerate(agents):
            agent_observable = False
            for sensor in sensors:
                agent_observable = agent_observable or sensor.in_range(agent_idx, t)

            if not agent_observable:
                continue

            expected_position = np.array([agent.x_true[t], agent.y_true[t], agent.z_true[t]])
            (noise_x, noise_y) = agent.measure(t)
            measured_position = expected_position + np.array([noise_x[0], noise_y[0], 0])

            o_s = Observation(agent_idx, measured_position, expected_position, agent.bimodal_evil, agent.bimodal_good)
            observe_set.append(o_s)
        return observe_set

class Agent:
    # make noise follow bimodal distribution with a peak at 0 for the good agent and the peak on the right for the bad agent
    pdf = stats.norm.pdf
    frac = 0.49
    # scale_factor_good = 2e-2
    # scale_factor_evil = 5e-2
    scale_factor_good = 1
    scale_factor_evil = 1

    # loc = mean, scale = stdev
    # set 1
    loc1, scale1, size1 = (0, 0.75, 110)
    loc2, scale2, size2 = (2, 0.25, int(frac*size1))
    loc3, scale3, size3 = (0, 0.25, 50)
    loc4, scale4, size4 = (2, 0.89, int(size3/frac))

    x_evil = np.concatenate([np.random.normal(loc=loc1, scale=scale1, size=size1),
                            np.random.normal(loc=loc2, scale=scale2, size=size2)])

    x_good = np.concatenate([np.random.normal(loc=loc3, scale=scale3, size=size3),
                            np.random.normal(loc=loc4, scale=scale4, size=size4)])
    def __init__(self, name, stk_ref, true_belief, times, x_true, y_true, z_true):
        self.name = name
        self.stk_ref = stk_ref
        self.true_belief = true_belief

        # times and positions calculated from STK's Data Providers in Earth Fixed Reference Frame
        self.times = times
        self.x_true = x_true
        self.y_true = y_true
        self.z_true = z_true

    def eval_evil_pdf(self, x_eval):
        bimodal_pdf = Agent.pdf(x_eval, loc=Agent.loc1, scale=Agent.scale1)  \
                    + Agent.pdf(x_eval, loc=Agent.loc2, scale=Agent.scale2)
        return bimodal_pdf

    def eval_good_pdf(self, x_eval):
        bimodal_pdf = Agent.pdf(x_eval, loc=Agent.loc3, scale=Agent.scale3)  \
                    + Agent.pdf(x_eval, loc=Agent.loc4, scale=Agent.scale4)
        return bimodal_pdf


    def intialize_bimodal_pdf(self):
        self.bimodal_evil_norm = sum([self.eval_evil_pdf(i) for i in np.linspace(-10,10,100)])
        self.bimodal_good_norm = sum([self.eval_good_pdf(i) for i in np.linspace(-10, 10, 100)])
        self.bimodal_evil = lambda x_eval: self.eval_evil_pdf(x_eval) / self.bimodal_evil_norm
        self.bimodal_good = lambda x_eval: self.eval_good_pdf(x_eval) / self.bimodal_good_norm

        # debug pdf plot
        # x1 = np.concatenate([np.random.normal(loc=Agent.loc1, scale=Agent.scale1, size=Agent.size1),
        #                      np.random.normal(loc=Agent.loc2, scale=Agent.scale2, size=Agent.size2)])
        # x_eval_1 = np.linspace(x1.min() - 1, x1.max() + 1, len(x1))
        #
        # x2 = np.concatenate([np.random.normal(loc=Agent.loc3, scale=Agent.scale3, size=Agent.size3),
        #                      np.random.normal(loc=Agent.loc4, scale=Agent.scale4, size=Agent.size4)])
        # x_eval_2 = np.linspace(x2.min() - 1, x2.max() + 1, len(x2))
        # plt.axis('off')
        # plt.plot(x_eval_2, self.bimodal_good(x_eval_2), 'b--', label="Good Agent PDF")
        # plt.plot(x_eval_1, self.bimodal_evil(x_eval_1), 'r--', label="Bad Agent PDF")
        # plt.legend()
        # plt.show()

        # pick one bimodal distribution if the agent is evil, pick the other if the are good
        if self.evil:
            self.bimodal_pdf = self.bimodal_evil
            self.scale_factor = Agent.scale_factor_evil
            self.x = Agent.x_evil
        else:
            self.bimodal_pdf = self.bimodal_good
            self.scale_factor = Agent.scale_factor_good
            self.x = Agent.x_good

    def measure(self, t_idx):
        x_eval = np.linspace(self.x.min() - 1, self.x.max() + 1, len(self.x))
        noise = self.scale_factor * np.random.choice(self.x, size=1, p=self.bimodal_pdf(x_eval)/np.sum(self.bimodal_pdf(x_eval)))
        
        if t_idx < len(self.times)-1:
            t_next = t_idx + 1
        else:
            # if we're at the end of the trajectory, consider the previous point to compute the tangent angle to trajectory
            t_next = t_idx - 1

        # angle of tangent line to trajectory
        theta = np.arctan2(self.y_true[t_next] - self.y_true[t_idx], self.x_true[t_next] - self.x_true[t_idx])

        noise_x = np.cos(theta) * noise
        noise_y = np.sin(theta) * noise

        return (noise_x, noise_y)

    def dumpcsv(self):
        print(self.name)
        np.savetxt("{name}.csv".format(name=self.name), (self.times, self.x_true, self.y_true, self.z_true), delimiter=',', fmt="%s")

    def plotinfo(self):   
        fig = plt.figure()
        ax = plt.axes()
        evil_label = "evil"
        theta = np.linspace(0, 2*np.pi, 150)
        radius = 0.175 # km
        plot = False

        # simulate good and bad noise in same interval
        if plot:
            if not self.evil:
                evil_label = "good"

            x_meas_good = []
            y_meas_good = []
            x_meas_evil = []
            y_meas_evil = []

            for i in range(2):
                for t_idx, t in enumerate(self.times):
                    (x_meas_t, y_meas_t) = self.measure(t_idx)  # generate noise
                    if self.interval[0] <= t <= self.interval[1]:  # seconds
                        if self.evil:
                            x_meas_evil.append(self.x_true[t_idx] + x_meas_t)
                            y_meas_evil.append(self.y_true[t_idx] + y_meas_t)
                        else:
                            x_meas_good.append(self.x_true[t_idx] + x_meas_t)
                            y_meas_good.append(self.y_true[t_idx] + y_meas_t)
                self.evil = not self.evil

                if self.evil:
                    self.scale_factor = 5e-2
                else:
                    self.scale_factor = 2e-2

            self.evil = not self.evil
            if self.evil:
                    self.scale_factor = 5e-2
            else:
                self.scale_factor = 2e-2

            ax.plot(self.x_true, self.y_true, label="true trajectory", color = "blue")
            ax.plot(x_meas_good, y_meas_good, label="good noise", color = "green")
            ax.plot(x_meas_evil, y_meas_evil, label="evil noise", color = "orange", alpha = 0.5)
            ax.plot(radius*np.cos(theta), radius*np.sin(theta), label="no fly zone", color = "red")
            plt.xlabel("X Position (km)")
            plt.ylabel("Y Position (km)")
            plt.title("XY Position for {0} {1} wrt Washington".format(evil_label, self.name))
            plt.legend()
            plt.show()

    def __str__(self):
        return str(self.name)

    # Sharing and Belief initializing structures
    def init_sharing_type(self,agent_list,async_flag=True,av_flag=False,no_bad=1):
        self.async_flag = async_flag
        self.av_flag = av_flag
        self.viewable_agents = []
        self.last_seen = {}
        self.agent_id = agent_list
        self.no_bad = 1
        self.evil = False
        self.initLastSeen(agent_list)

    def init_belief(self,belief_size):
        no_system_states = 2 ** belief_size
        belief_value = 1.0 / no_system_states
        base_list = [0, 1]  # 0 is bad, 1 is good
        total_list = [base_list for n in range(belief_size)]
        # self.id_idx = dict([[k, j] for j, k in enumerate(agent_id)])
        self.local_belief = {}
        self.diff_belief = {}
        self.belief_bad = []
        self.belief_calls = 0
        for t_p in itertools.product(*total_list):
            self.local_belief.update({t_p: belief_value})
        self.actual_belief = deepcopy(self.local_belief)
        if abs(sum(self.local_belief.values()) - 1.0) > 1e-6:
            raise ProbablilityNotOne("Sum is " + str(sum(self.local_belief.values())))

        if self.async_flag:
            self.neighbor_set = dict()
            self.neighbor_belief = dict()
            self.resetFlags = dict()
            for t_p in self.local_belief:
                self.neighbor_set[t_p] = set()
                self.neighbor_belief[t_p] = dict()
                self.resetFlags[t_p] = True
                for n_i in self.agent_id:
                    self.neighbor_belief[t_p][n_i] = -1  ## b^a_j(theta)

    # Last seen for asynchronous ADHT
    def initLastSeen(self, agent_id):
        for a_i in agent_id:
            self.last_seen.update({a_i: 0})

    def resetBelief(self, belief, belief_arrays):
        for j in self.neighbor_belief[belief]:
            if j is not self.name:
                self.neighbor_belief[belief][j] = -1
        for k in self.local_belief:
            if k is not belief:
                self.neighbor_set[k] = set([self.name])
        self.resetFlags[belief] = False

    def asyncBeliefUpdate(self, belief, belief_arrays):
        if self.resetFlags[belief]:
            self.resetBelief(belief, belief_arrays)
        for j in belief_arrays:
            for t_p in [kj for kj in belief_arrays[j] if kj is not belief]:  ## t_p is theta_prime
                if self.neighbor_belief[belief][j] == -1:  ##j is in
                    self.neighbor_set[t_p].add(j)
            self.neighbor_belief[belief][j] = belief_arrays[j][belief]  ## Not sure what line 7 does??
        for t_p in [kj for kj in self.local_belief if kj is not belief]:
            if len(self.neighbor_set[t_p]) < 2 * self.no_bad + 1:
                return False
        self.resetFlags[belief] = True
        return True

    def ADHT(self, belief_arrays):
        actual_belief = {}
        neighbor_set = {}
        for theta in self.actual_belief:
            if self.asyncBeliefUpdate(theta, belief_arrays):
                self.belief_calls += 1
                space = set.union(*[self.neighbor_set[x] for x in self.neighbor_set if x is not theta])
                belief_list = []
                agent_order = []
                for j in space:
                    belief_list.append(self.neighbor_belief[theta][j])
                    agent_order.append(j)
                belief_list, agent_order = zip(*sorted(zip(belief_list, agent_order)))
                neighbor_set[theta] = set(agent_order[self.no_bad:])
                if self.av_flag:
                    actual_belief[theta] = min(self.local_belief[theta],
                                               np.mean(list(belief_list)[self.no_bad:-1 * self.no_bad]))
                else:
                    actual_belief[theta] = min([self.local_belief[theta]] + list(belief_list)[self.no_bad:])
            else:
                actual_belief.update({theta: min(self.actual_belief[theta], self.local_belief[theta])})
            if actual_belief[theta] < 0:
                print('Negative')
        self.actual_belief = dict(
            [[theta, actual_belief[theta] / sum(actual_belief.values())] for theta in actual_belief])
        for n_s in neighbor_set:
            self.neighbor_set[n_s] = neighbor_set[n_s]
        if self.evil:
            # random_belief = np.random.rand(len(self.local_belief))
            # random_belief /= np.sum(random_belief)
            for b_i in self.local_belief:
                self.local_belief[b_i] = 0
                self.actual_belief[b_i] = 0
            # new
            self.local_belief[(1,0,1,1,0)] = 1
            self.actual_belief[(1, 0, 1, 1, 0)] = 1

    def shareBelief(self, belief_arrays):
        actual_belief = {}
        if len(belief_arrays) >= 2 * self.no_bad + 1:  ## Case 1
            actual_belief = dict()
            for theta in self.actual_belief:
                self.belief_calls += 1
                sorted_belief = sorted([b_a[theta] for b_a in belief_arrays.values()], reverse=True)
                for f in range(self.no_bad):
                    sorted_belief.pop()
                    if self.av_flag:
                        sorted_belief.pop(-1)
                if self.av_flag:
                    actual_belief.update({theta: min(self.local_belief[theta], np.mean(sorted_belief))})
                else:
                    sorted_belief.append(self.local_belief[theta])
                    actual_belief.update({theta: min(sorted_belief)})  # Minimum
        else:  # Case 2
            for theta in self.actual_belief:
                actual_belief.update({theta: min(self.actual_belief[theta], self.local_belief[theta])})
        # Normalize
        self.actual_belief = dict(
            [[theta, actual_belief[theta] / sum(actual_belief.values())] for theta in actual_belief])
        if self.evil:
            random_belief = np.random.rand(len(self.local_belief))
            random_belief /= np.sum(random_belief)
            for b_i, r_b in zip(self.local_belief, random_belief):
                self.local_belief[b_i] = r_b
                self.actual_belief[b_i] = r_b
        sys_t = list(self.actual_belief.keys())[np.argmax(list(self.actual_belief.values()))]
        for i, s_i in enumerate(sys_t):
            if s_i == 0:
                self.belief_bad.append(i)

    # Local Observation
    def observe(self,observe_set):
        obs_list = []
        for o_s in observe_set:
            observed_diff = o_s.measured - o_s.expected_position
            noise_measure = np.sqrt(observed_diff[0]**2+observed_diff[1]**2)
            probability_bad = o_s.bimodal_bad(noise_measure)
            probability_good = o_s.bimodal_good(noise_measure)
            obs_list.append((o_s.idx,probability_bad,probability_good))
        return obs_list
        #
        #
        #
        #
        # # guess = np.zeros((len(self.agent_id),1))
        # idx = np.random.randint(0,5)
        # # for i in range(len(self.agent_id)):
        # accuracy = 0.85
        # sigma = 0.05
        # if idx == 4:
        #     s = np.random.normal(1-accuracy, sigma,1)
        #     s = np.clip(s,0,1)
        # else:
        #     s = np.random.normal(accuracy, sigma,1)
        #     s = np.clip(s,0,1)
        # # guess[i] = s
        # return idx,s

    def likelihood(self, sys_status,observation_list):
        epsilon = 1e-9

        # Work through each element of the tuple, if is likely then good if its unlikely then bad.
        # view_index = [self.id_idx[v_a] for v_a in viewable_agents]
        prob_i = 1.0
        for o_s in observation_list:
            if sys_status[o_s[0]] == 0:
                prob_i *= o_s[1] + epsilon
            else:
                prob_i *= o_s[2] + epsilon
        # for v_i, v_p in zip([observation[0]],[observation[1]]):
        #     if len(v_p) > 1:
        #         if sys_status[v_i] == 0:
        #             prob_i *= v_p[1]  # Probability for bad model
        #         else:
        #             prob_i *= v_p[0]  # Probability for good model
        #     else:
        #         if sys_status[v_i] == 0:  # if bad
        #             prob_i *= 1.0 - v_p + epsilon
        #         else:
        #             prob_i *= v_p + epsilon
        return prob_i

    def updateLocalBelief(self, viewable_agents=None):
        ## Synchronous update rule
        tot_b = 0.0
        observation = self.observe(viewable_agents)
        for b_i in self.local_belief:
            tot_b += self.likelihood(b_i,observation)*self.local_belief[b_i]
        for b_i in self.local_belief:
            self.local_belief[b_i] = self.likelihood(b_i,observation)*self.local_belief[b_i] / tot_b
        if abs(sum(self.local_belief.values()) - 1.0) > 1e-6:
            raise ProbablilityNotOne("Sum is " + str(sum(self.local_belief.values())))
        if self.evil:
            for b_i in self.local_belief:
                self.local_belief[b_i] = 0
                self.actual_belief[b_i] = 0
            random_belief = np.random.rand(len(self.local_belief))
            random_belief /= np.sum(random_belief)
            for b_i, r_b in zip(self.local_belief, random_belief):
                self.local_belief[b_i] = r_b
                self.actual_belief[b_i] = r_b
    ##TODO: Realistic observation functions from STK

    def draw_belief_graph(self,fig_dim,run_length,color):
        self.fig = plt.figure(figsize=(1.6125*fig_dim, fig_dim))
        self.ax = plt.subplot(111)
        self.ax.set_xlim([0, run_length])
        self.ax.set_ylim([0,1.1])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.x_data = [0]
        self.y_data = [0.03125]
        self.belief_line = self.ax.plot(self.x_data,self.y_data, linewidth=5, color=color, zorder=1)
        self.ax.set_xticks([0,int(run_length/2),run_length])
        self.ax.set_yticks([0, 0.5,1])
        self.ax.tick_params(axis='both', which='major', labelsize=18,direction="in")


    def update_graph(self,i_no,belief):
        self.x_data.append(i_no)
        self.y_data.append(belief)
        self.belief_line[0].set_data(self.x_data,self.y_data)

    def save_belief(self,exp_name,idx,i_no):
        self.fig.savefig(os.path.join(os.getcwd(),'Images',f'{exp_name}','Belief_Graph',f'Agent_{idx}',f'Belief_{i_no}.png'),transparent=True)