from policy import Policy
from mdp import *
import math
import itertools
import os
import subprocess
from scipy.special import comb
import numpy as np
from copy import deepcopy
import random
import operator
from collections import OrderedDict
import pickle
import policy_reactive
import simplejson as json
import operator


class ProbablilityNotOne(Exception):
	pass


class Agent():
	def __init__(self, init=None, target_list=[], meeting_state=[], gw_env=None, belief_tracks=None, id_no=None,
				 policy_load=False, slugs_location=None, evil=False,trace_load=False):
		self.id_no = id_no
		self.init = init
		self.current = init
		self.alpha = 1.00
		self.burn_rate = 1.00
		self.targets = list(target_list.keys())
		self.target_dict = target_list
		self.meeting_state = meeting_state
		self.belief_tracks = belief_tracks
		self.evil = evil
		self.async_flag = True
		self.av_flag = False
		self.belief_calls = 0
		self.gw = gw_env
		self.viewable_agents = []
		self.information = dict([[m, set()] for m in range(len(target_list))])
		self.comms_env = [0, ] * len(target_list)
		self.last_seen = {}
		self.error_prob = 0.001
		self.policy_load = policy_load
		self.slugs_loc = slugs_location
		self.steps = 0
		self.stop_val = 0.95

		## Triggers for policy
		self.current_node = 0
		no_targets = len(self.targets)
		self.current_env = (0,) * no_targets
		self.env_list = list(itertools.product(*[(0, 1)] * (no_targets + 1)))
		self.env_list_conv = list(itertools.product(*[(0, 1)] * (no_targets)))
		self.conv = 0
		self.state_label = 's'
		self.trace_load = trace_load

	def reset(self):
		self.information = dict([[m, set()] for m in range(len(self.target_dict))])
		self.comms_env = [0, ] * len(self.target_dict)
		self.current = self.init
		self.viewable_agents = []
		self.current_node = 0
		no_targets = len(self.targets)
		self.current_env = (0,) * no_targets
		self.env_list = list(itertools.product(*[(0, 1)] * (no_targets + 1)))
		self.env_list_conv = list(itertools.product(*[(0, 1)] * (no_targets)))
		self.initBelief(list(self.id_idx.keys()),1,len(self.targets))
		self.belief_calls=0

	# self.savePolicy()
	def initInfo(self, agent_loc):
		self.informationDict = dict([[k_i, deepcopy(self.information)] for k_i in agent_loc])

	def update(self, agent_loc):
		# if self.conv == 1:
		# 	self.policy[self.current_node]['Successors'][self.env_list_conv.index(self.current_env)]
		# else:
		if self.trace_load:
			s = int(self.trace_load.pop(0))
		else:
			env_ind = np.nonzero(1-np.array(list(self.policy[self.current_node]['State'].values())[0:-1]))[0]
			if 2**len(env_ind)>len(self.policy[self.current_node]['Successors']) or len(env_ind)==0:
				self.current_node = self.policy[self.current_node]['Successors'][0]
			else:
				# print('Current Env {}       Env index: {}'.format(self.current_env, env_ind))
				self.current_node = self.policy[self.current_node]['Successors'][int(''.join(map(str,np.array(list(self.current_env + (self.conv,)))[env_ind].tolist())),2)]
			# self.current_node = self.policy[self.current_node]['Successors'][self.env_list.index(self.current_env + (self.conv,))]
			s = self.policy[self.current_node]['State'][self.state_label]
		self.updateAgent(s)
		## Vision
		self.updateVision(self.current, agent_loc)
		self.current_env = (0,) * len(self.targets)
		info_len = [len(self.information[k]) for k in self.information]
		if s in self.meeting_state and self.conv ==0:
			self.conv = 1
		elif self.conv == 1 and max(info_len)>1:
			self.conv = 0
		self.updateConvFlags(agent_loc)

		return self.current

	def updateConvFlags(self, agent_loc):
		# Inserting information if passes over a target
		for a_i, a_l in agent_loc.items():
			if a_l in self.targets:
				self.informationDict[a_i][self.targets.index(a_l)].add(a_i)
				if a_i == self.id_no:
					self.information[self.targets.index(a_l)].add(a_i)
		for a_i, a_l in agent_loc.items():
			view_agents = self.agent_in_view(a_l, agent_loc.values(), agent_loc.keys())[0]
			for v_a in view_agents:
				for k in self.informationDict[a_i]:
					self.informationDict[a_i][k].update(self.informationDict[v_a][k])
					if a_i == self.id_no:
						self.information[k].update(self.informationDict[v_a][k])
		for t_i in self.information:
			if len(self.information[t_i]) >= 2 * self.no_bad + 1:
				hold_env = list(self.current_env)
				hold_env[t_i] = 1
				self.current_env = tuple(hold_env)
				self.information[t_i].clear()

	def writeOutputTimeStamp(self, init=[]):
		out_dict = dict()
		out_dict.update({'AgentLoc': self.current})
		out_dict.update({'ActBelief': deepcopy(self.actual_belief)})
		out_dict.update({'LastSeen': deepcopy(self.last_seen)})
		out_dict.update({'Visible': self.viewable_agents})
		# out_dict.update({'NominalTrace': self.policy.nom_trace})
		# out_dict.update({'BadTrace': self.policy.bad_trace})
		out_dict.update({'BeliefCalls': self.belief_calls})
		if init:
			out_dict.update({'Targets': self.targets})
			out_dict.update({'Id_no': list(init)})
			out_dict.update({'BadBelief': self.belief_tracks[0]})
			out_dict.update({'GoodBelief': self.belief_tracks[1]})
			out_dict.update({'Rows': self.gw.nrows})
			out_dict.update({'Cols': self.gw.ncols})
		return out_dict

	def Policy(self, infile=None, slugs_location=None, preload=False,meeting=False):

		if infile == None:
			infile = 'grid_example_{}'.format(self.id_no)
			filename = infile + '.structuredslugs'
		else:
			filename = infile + '_{}'.format(self.id_no) + '.structuredslugs'

		if preload:
			return self.parseJson(infile + '.json')

		file = open(filename, 'w')

		file.write('[INPUT]\n')
		for t in self.targets:
			file.write('c{}:0...1\n'.format(t))
		file.write('shared:0...1\n')
		file.write('\n')  # Add moving obstacles

		file.write('\n[OUTPUT]\n')
		file.write('s:0...{}'.format(self.gw.nstates - 1))

		file.write('\n[ENV_INIT]\n')
		for t in self.targets:
			file.write('c{} = 0\n'.format(t))
		file.write('shared = 0\n')
		file.write('\n')  # Add moving obstacles

		file.write('\n[SYS_INIT]\n')
		file.write('s={}\n'.format(self.current))

		file.write('\n[ENV_TRANS]\n')
		file.write('shared=1 -> shared\'=0\n')
		for t in self.targets:
			file.write('c{} = 1 -> c{}\' = 0\n'.format(t, t))

		if not meeting:
			# """# # No Meeting #  Comment/Uncomment Here
			file.write('\n[SYS_TRANS]\n')

			for s in self.gw.states:
				strn = 's = {} ->'.format(s)
				repeat = set()
				for u in self.gw.actlist:
					snext = np.nonzero(self.gw.prob[u][s])[0][0]
					if snext not in repeat:
						repeat.add(snext)
						strn += 's\' = {} \\/ '.format(snext)
				strn = strn[:-3]
				file.write(strn + '\n')
			for s in self.gw.obstacles:
				file.write('!s = {}\n'.format(s))
			file.write('\n[SYS_LIVENESS]\n')
			t_s = self.id_idx[self.id_no] % len(self.targets)

			for i,s in enumerate(self.targets[t_s:]+self.targets[0:t_s]):
				file.write('s = {} \n'.format(s))
			#"""

		""" Meeting Comment/Uncomment Here
		file.write('\n[ENV_LIVENESS]\n')
		str = ''
		for t in self.meeting_state:
			str += 's={} \\/'.format(t)
		str = str[:-3]
		str += '-> shared = 1\n'
		file.write(str)

		file.write('\n[SYS_TRANS]\n')

		for s in self.gw.states:
			strn = 's = {} ->'.format(s)
			repeat = set()
			for u in self.gw.actlist:
				snext = np.nonzero(self.gw.prob[u][s])[0][0]
				if snext not in repeat:
					repeat.add(snext)
					strn += 's\' = {} \\/ '.format(snext)
			strn = strn[:-3]
			file.write(strn + '\n')
		for s in self.gw.obstacles:
			file.write('!s = {}\n'.format(s))
		file.write('\n[SYS_LIVENESS]\n')
		t_s = self.id_idx[self.id_no] % len(self.targets)


		for i, s in enumerate(self.targets[t_s:] + self.targets[0:t_s]):
			if i == 0:
				file.write('s = {} \n'.format(s))
				# file.write('s = {} \\/ c{} = {}\n'.format(s, s, 1))
			else:
				file.write('s = {} \\/ c{} = {}\n'.format(s, s, 1))
				# file.write('s = {} \n'.format(s))
		file.write('s = {}\n'.format(self.meeting_state[0]))
		file.write('shared = 1')
		#"""
		if meeting:
			# """ Meeting2 Comment/Uncomment Here
			file.write('\n[ENV_LIVENESS]\n')
			str = ''
			for t in self.meeting_state:
				str += 's={} \\/'.format(t)
			str = str[:-3]
			str += '-> shared = 1\n'
			file.write(str)

			file.write('\n[SYS_TRANS]\n')

			for s in self.gw.states:
				strn = 's = {} ->'.format(s)
				repeat = set()
				for u in self.gw.actlist:
					snext = np.nonzero(self.gw.prob[u][s])[0][0]
					if snext not in repeat:
						repeat.add(snext)
						strn += 's\' = {} \\/ '.format(snext)
				strn = strn[:-3]
				file.write(strn + '\n')
			for s in self.gw.obstacles:
				file.write('!s = {}\n'.format(s))
			file.write('\n[SYS_LIVENESS]\n')
			t_s = self.id_idx[self.id_no] % len(self.targets)


			for i, s in enumerate(self.targets[t_s:] + self.targets[0:t_s]):
				if i == 0:
					file.write('s = {} \n'.format(s))
					# file.write('s = {} \\/ c{} = {}\n'.format(s, s, 1))
				else:
					# file.write('s = {} \\/ c{} = {}\n'.format(s, s, 1))
					file.write('s = {} \n'.format(s))
			file.write('s = {}\n'.format(self.meeting_state[0]))
			file.write('shared = 1')
			#"""

		file.close()

		if slugs_location != None:
			os.system(
				'python2 ' + slugs_location + 'tools/StructuredSlugsParser/compiler.py ' + infile + '.structuredslugs > ' + infile + '.slugsin')
			sp = subprocess.Popen(
				slugs_location + 'src/slugs --explicitStrategy --jsonOutput ' + infile + '.slugsin > ' + infile + '.json',
				shell=True, stdout=subprocess.PIPE)
			sp.wait()
			print('Computing controller...')
			return self.parseJson(infile + '.json')

	def writeJson(self, infile, outfile, dict=None):
		if dict is None:
			dict = self.parseJson(infile)
		j = json.dumps(dict, indent=1)
		f = open(outfile, 'w')
		print >> f, j
		f.close()

	def parseJson(self, filename, outfilename=None):
		automaton = dict()
		file = open(filename)
		data = json.load(file)
		file.close()
		variables = dict()
		for var in data['variables']:
			v = var.split('@')[0]
			if v not in variables.keys():
				for var2ind in range(data['variables'].index(var), len(data['variables'])):
					var2 = data['variables'][var2ind]
					if v != var2.split('@')[0]:
						variables[v] = [data['variables'].index(var), data['variables'].index(var2)]
						break
					if data['variables'].index(var2) == len(data['variables']) - 1:
						variables[v] = [data['variables'].index(var), data['variables'].index(var2) + 1]

		for s in data['nodes'].keys():
			automaton[int(s)] = dict.fromkeys(['State', 'Successors'])
			automaton[int(s)]['State'] = dict()
			automaton[int(s)]['Successors'] = []
			for v in variables.keys():
				if variables[v][0] == variables[v][1]:
					bin = [data['nodes'][s]['state'][variables[v][0]]]
				else:
					bin = data['nodes'][s]['state'][variables[v][0]:variables[v][1]]
				automaton[int(s)]['State'][v] = int(''.join(str(e) for e in bin)[::-1], 2)
				automaton[int(s)]['Successors'] = data['nodes'][s]['trans']
		if outfilename == None:
			return automaton
		else:
			self.writeJson(None, outfilename, automaton)
			return automaton

	def definePolicyDict(self, id_list, policy_array):
		self.policy_list = dict([[i, p_i] for i, p_i in zip(id_list, policy_array)])

	def savePolicy(self):
		print(self.id_no)
		with open('policies/' + str(self.id_no) + '.pkl', 'wb') as f:
			pickle.dump(self.policy, f, pickle.HIGHEST_PROTOCOL)

	def loadPolicy(self):
		print(self.id_no)
		with open('policies/' + str(self.id_no) + '.pkl', 'rb') as f:
			self.policy = pickle.load(f)

	def updateAgent(self, state):
		self.steps += 1 if state != self.current else 0
		self.current = state

	### Belief Rules
	def initBelief(self, agent_id, no_bad, no_targets):
		self.no_bad = no_bad
		no_agents = len(agent_id)
		## Combinarotic approach
		# no_system_states = 0
		# for b_d in range(no_bad+1):
		# 	no_system_states += comb(no_agents,b_d)
		# ## Unknown number of bad agents
		no_system_states = 2 ** no_targets
		belief_value = 1.0 / no_system_states
		base_list = [0, 1]  # 0 is bad, 1 is good
		total_list = [base_list for n in range(no_targets)]
		self.id_idx = dict([[k, j] for j, k in enumerate(agent_id)])
		self.local_belief = {}
		self.diff_belief = {}
		self.belief_bad = []
		self.belief_bad = []
		for t_p in itertools.product(*total_list):
			# ## Combinartoic approach
			# if sum(t_p) >= no_agents-no_bad:
			# 	self.local_belief.update({t_p:belief_value})
			# ## Unknown number of bad
			self.local_belief.update({t_p: belief_value})
			self.diff_belief.update({t_p: set([t_p])})
		self.actual_belief = deepcopy(self.local_belief)
		if abs(sum(self.local_belief.values()) - 1.0) > 1e-6:
			raise ProbablilityNotOne(
				"Sum is " + str(sum(self.local_belief.values())))  # ,"Sum is "+str(sum(self.local_belief.values())))
		if self.async_flag:
			self.neighbor_set = dict()
			self.neighbor_belief = dict()
			self.resetFlags = dict()
			for t_p in self.local_belief:
				self.neighbor_set[t_p] = set()
				self.neighbor_belief[t_p] = dict()
				self.resetFlags[t_p] = True
				for n_i in agent_id:
					self.neighbor_belief[t_p][n_i] = -1  ## b^a_j(theta)

	def initPolicy(self, pre_load=False,meeting=False):
		if self.policy_load:
			self.loadPolicy()
		else:
			if pre_load:
				self.policy = self.Policy(slugs_location=self.slugs_loc, preload=True)
			else:
				self.policy = self.Policy(slugs_location=self.slugs_loc,meeting=meeting)

	def updateBelief(self, viewable_agents, target):
		if not target:
			return
		## Synchronous update rule
		tot_b = 0.0
		self.alpha *= self.burn_rate
		self.belief_bad = []
		# for b_i in self.local_belief:
		# 	tot_b += self.likelihood(b_i, viewable_agents,target)*self.local_belief[b_i]
		for b_i in self.local_belief:
			self.local_belief[b_i] = (1 - self.alpha) * self.local_belief[b_i] + self.alpha * self.likelihood(b_i,
																											  viewable_agents,
																											  target[
																												  0]) * \
									 self.local_belief[b_i]
			for b_z in self.local_belief:
				if b_i[target[1]] != b_z[target[1]]:
					self.diff_belief[b_i].add(b_z)
		tot_b = sum(self.local_belief.values())
		for b_i in self.local_belief:
			self.local_belief[b_i] /= tot_b

		if abs(sum(self.local_belief.values()) - 1.0) > 1e-6:
			raise ProbablilityNotOne("Sum is " + str(sum(self.local_belief.values())))
		if self.evil:
			for b_i in self.local_belief:
				self.local_belief[b_i] = 0.0
				self.actual_belief[b_i] = 0.0
			if isinstance(self.evil, tuple):
				for b_i in self.local_belief:
					if b_i == self.evil:
						self.local_belief[b_i] = 1.0
						self.actual_belief[b_i] = 1.0
					for b_z in self.local_belief:
						self.diff_belief[b_i].add(b_z)
			else:
				random_belief = np.random.rand(len(self.local_belief))
				random_belief /= np.sum(random_belief)
				for b_i, r_b in zip(self.local_belief, random_belief):
					self.local_belief[b_i] = r_b
					self.actual_belief[b_i] = r_b
					for b_z in self.local_belief:
						self.diff_belief[b_i].add(b_z)

	def updateVision(self, state, agent_states):
		self.viewable_agents, viewable_states = self.agent_in_view(state, agent_states.values(), agent_states.keys())
		# viewable_states = [self.observation(agent_states[v_s]) for v_s in self.viewable_agents]
		self.updateBelief(self.viewable_agents, viewable_states)

	def shareBelief(self, belief_arrays):
		actual_belief = {}
		if len(belief_arrays) >= 2 * self.no_bad + 1:  ## Case 1
			actual_belief = dict()
			for theta in self.actual_belief:
				self.belief_calls += 1
				sorted_belief = sorted([b_a[theta] for b_a in belief_arrays], reverse=True)
				for f in range(self.no_bad):
					sorted_belief.pop()
					if self.av_flag:
						sorted_belief.pop(-1)
				if self.av_flag:
					actual_belief.update({theta: min(self.local_belief[theta], np.mean(sorted_belief))})
				else:
					sorted_belief.append(self.local_belief[theta])
					actual_belief.update({theta: min(sorted_belief)})  # Minimum
			# actual_belief.update({theta:np.average(sorted_belief)}) #Averaging
		else:  # Case 2
			for theta in self.actual_belief:
				actual_belief.update({theta: min(self.actual_belief[theta], self.local_belief[theta])})
			# if self.av_flag:
			# 	actual_belief.update({theta:np.mean([self.actual_belief[theta]]+[self.local_belief[theta]])})
			# else:
		# Normalize
		self.actual_belief = dict(
			[[theta, actual_belief[theta] / sum(actual_belief.values())] for theta in actual_belief])
		if self.evil:
			for b_i in self.local_belief:
				self.local_belief[b_i] = 0.0
				self.actual_belief[b_i] = 0.0
			if isinstance(self.evil, tuple):
				for b_i in self.local_belief:
					if b_i == self.evil:
						self.local_belief[b_i] = 1.0
						self.actual_belief[b_i] = 1.0
					for b_z in self.local_belief:
						self.diff_belief[b_i].add(b_z)
			else:
				random_belief = np.random.rand(len(self.local_belief))
				random_belief /= np.sum(random_belief)
				for b_i, r_b in zip(self.local_belief, random_belief):
					self.local_belief[b_i] = r_b
					self.actual_belief[b_i] = r_b
					for b_z in self.local_belief:
						self.diff_belief[b_i].add(b_z)
		sys_t = list(self.actual_belief.keys())[np.argmax(list(self.actual_belief.values()))]
		for i, s_i in enumerate(sys_t):
			if s_i == 0:
				self.belief_bad.append(i)

	def ADHT(self, belief_arrays, info_packet):
		global belief_calls
		actual_belief = {}
		neighbor_set = {}
		for theta in self.actual_belief:
			if self.asyncBeliefUpdate(theta, belief_arrays):
				# for t_i in self.information:
				# 	for v_i in [j_i for j_i in self.neighbor_belief[theta] if self.neighbor_belief[theta][j_i] != -1]:
				# 		self.information[t_i] = self.information[t_i].union(info_packet[v_i][t_i])
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
					actual_belief[theta] = min([self.local_belief[theta]] + list(belief_list)[self.no_bad:-1 * self.no_bad])
					# actual_belief[theta] = min(list(belief_list)[self.no_bad:-1 * self.no_bad])
			else:
				if self.observation():
					actual_belief.update({theta: min(self.actual_belief[theta], self.local_belief[theta])})
					# actual_belief.update({theta:self.actual_belief[theta]})
				else:
					actual_belief.update({theta: self.actual_belief[theta]})
			if actual_belief[theta] < 0:
				print('Negative')
		self.actual_belief = dict(
			[[theta, actual_belief[theta] / sum(actual_belief.values())] for theta in actual_belief])
		for n_s in neighbor_set:
			self.neighbor_set[n_s] = neighbor_set[n_s]
		if self.evil:
			for b_i in self.local_belief:
				self.local_belief[b_i] = 0.0
				self.actual_belief[b_i] = 0.0
			if isinstance(self.evil, tuple):
				for b_i in self.local_belief:
					if b_i == self.evil:
						self.local_belief[b_i] = 1.0
						self.actual_belief[b_i] = 1.0
					for b_z in self.local_belief:
						self.diff_belief[b_i].add(b_z)
			else:
				random_belief = np.random.rand(len(self.local_belief))
				random_belief /= np.sum(random_belief)
				for b_i, r_b in zip(self.local_belief, random_belief):
					self.local_belief[b_i] = r_b
					self.actual_belief[b_i] = r_b
					for b_z in self.local_belief:
						self.diff_belief[b_i].add(b_z)
		for t_i in self.information:
			if len(self.information[t_i]) >= 2 * self.no_bad + 1:
				self.comms_env[t_i] = 1
			# self.information[t_i] = set()
			if self.comms_env[t_i] == 1 and self.current in self.meeting_state:
				self.comms_env[t_i] = 0

	def asyncBeliefUpdate(self, belief, belief_arrays):
		if self.resetFlags[belief]:
			self.resetBelief(belief, belief_arrays)
		for j in belief_arrays[belief]:
			for t_p in [kj for kj in belief_arrays[belief][j] if kj is not belief]:  ## t_p is theta_prime
				if self.neighbor_belief[belief][j] == -1:  ##j is in
					self.neighbor_set[t_p].add(j)
				# self.neighbor_belief[belief][j] = belief_arrays[belief][j][belief]
		for j in belief_arrays[belief]:
			self.neighbor_belief[belief][j] = belief_arrays[belief][j][belief]  ## Not sure what line 7 does??
		for t_p in [kj for kj in self.local_belief if kj is not belief]:
			if len(self.neighbor_set[t_p]) < 2 * self.no_bad + 1:
				return False
		self.resetFlags[belief] = True
		return True

	def resetBelief(self, belief, belief_arrays):
		for j in self.neighbor_belief[belief]:
			if j is not self.id_no:
				self.neighbor_belief[belief][j] = -1
		for k in self.local_belief:
			if k is not belief:
				self.neighbor_set[k] = set([self.id_no])
		self.resetFlags[belief] = False

	def likelihood(self, sys_status, viewable_agents, target):
		# Work through each element of the tuple, if is likely then good if its unlikely then bad.
		target_ind = self.targets.index(self.current)
		if sys_status[target_ind] == target:
			return self.target_dict[self.current] if target == 1 else 1 - self.target_dict[self.current]
		else:
			return 1 - self.target_dict[self.current] if target == 1 else self.target_dict[self.current]

	def observation(self, rtn_target=False):
		if self.current in self.targets:
			if rtn_target:
				return (
				np.random.choice([0, 1], 1, p=[1 - self.target_dict[self.current], self.target_dict[self.current]])[0],
				self.targets.index(self.current))
			else:
				return np.random.choice([0, 1], 1,
										p=[1 - self.target_dict[self.current], self.target_dict[self.current]])

		return None

	# ##### Vision rules
	def updateVision(self, state, agent_states):
		self.viewable_agents, viewable_states = self.agent_in_view(state, agent_states.values(), agent_states.keys())
		target_value = self.observation(True)
		self.updateTime()
		self.updateBelief(self.viewable_agents, target_value)
		self.updateLastSeen(self.viewable_agents, viewable_states)

	#
	def agent_in_view(self, state, agent_states, agent_id):
		view_agents = []
		view_states = []
		for a_s, a_i in zip(agent_states, agent_id):
			if isinstance(a_s, list):
				if a_s[0] in self.gw.observable_states[state[0]]:
					view_agents.append(a_i)
					obs_states = list(self.gw.observable_states[state[0]])
					view_states.append((int(np.random.choice(tuple(self.gw.observable_states[state[0]]),
															 p=self.observation(obs_states, a_s[0], self.error_prob))),
										a_s[1]))
			else:
				if a_s in self.gw.observable_states[state]:
					view_agents.append(a_i)
					obs_states = list(self.gw.observable_states[state])
					view_states.append(a_s)
		return view_agents, view_states

	#
	def initLastSeen(self, agent_id, agent_states):
		for a_s, a_i in zip(agent_states, agent_id):
			self.last_seen.update({a_i: [a_s, 0]})  # dictionary of lists: [state,time since observed in that state]

	#
	def updateLastSeen(self, agent_id, agent_states):
		assert len(agent_id) == len(agent_states)
		for a_i, a_s in zip(agent_id, agent_states):
			self.last_seen[a_i] = [a_s, 0]

	#
	def updateTime(self):
		for l_i in self.last_seen:
			self.last_seen[l_i][1] += 1

