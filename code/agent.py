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


class ProbablilityNotOne(Exception):
	pass

class Agent():
	def __init__(self,init=None,target_list=[],meeting_state = [],gw_env=None,belief_tracks=None,id_no=None,policy_load=False,slugs_location=None,evil=False):
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
		# self.async_flag = False
		# t_num = list(range(len(target_list)))
		# t_list = list(zip(self.targets,t_num))#[t_num[-1]]+t_num[:-1]))
		# p_list = list(zip(self.public_targets,t_num))#[t_num[-1]]+t_num[:-1]))
		# b_list = list(zip(self.bad_model,t_num))
		self.gw = gw_env
		self.viewable_agents = []
		self.last_seen = {}
		self.error_prob = 0.2

		if policy_load:
			self.loadPolicy()
		else:
			self.policy = self.Policy(slugs_location=slugs_location)
			# self.savePolicy()

	def writeOutputTimeStamp(self,init=[]):
		out_dict = dict()
		out_dict.update({'AgentLoc': self.current})
		out_dict.update({'ActBelief': deepcopy(self.actual_belief)})
		out_dict.update({'LastSeen': deepcopy(self.last_seen)})
		out_dict.update({'Visible': self.viewable_agents})
		# out_dict.update({'NominalTrace': self.policy.nom_trace})
		# out_dict.update({'BadTrace': self.policy.bad_trace})
		out_dict.update({'BeliefCalls':self.belief_calls})
		if init:
			out_dict.update({'Targets': self.targets})
			out_dict.update({'Id_no': list(init)})
			out_dict.update({'BadBelief':self.belief_tracks[0]})
			out_dict.update({'GoodBelief':self.belief_tracks[1]})
		return out_dict


	def Policy(self,infile=None,slugs_location=None):

		if infile == None:
			infile = 'grid_example'
			filename = infile+'.structuredslugs'
		else:
			filename = infile+'.structuredslugs'

		file = open(filename, 'w')


		file.write('[INPUT]\n')
		for t in self.targets:
			file.write('c{}:0...1\n'.format(t))
		file.write('\n') # Add moving obstacles

		file.write('\n[OUTPUT]\n')
		file.write('s:0...{}'.format(self.gw.nstates-1))

		file.write('\n[ENV_INIT]\n')
		for t in self.targets:
			file.write('c{} = 0\n'.format(t))
		file.write('\n') # Add moving obstacles

		file.write('\n[SYS_INIT]\n')
		file.write('s={}\n'.format(self.current))

		file.write('\n[ENV_LIVENESS]\n')
		for s in self.targets:
			str = ''
			for t in self.meeting_state:
				str += 's={} \\/'.format(t)
			str = str[:-3]
			str += '-> c{} = 1\n'.format(s)
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
			file.write(strn+'\n')
		for s in self.gw.obstacles:
			file.write('!s = {}\n'.format(s))
		file.write('\n[SYS_LIVENESS]\n')
		for s in self.targets:
			file.write('s = {} \\/ c{} = {} \n'.format(s,t,1))

		file.close()

		if slugs_location!=None:
			os.system(
			'python ' + slugs_location + 'tools/StructuredSlugsParser/compiler.py ' + infile + '.structuredslugs > ' + infile + '.slugsin')
			sp = subprocess.Popen(slugs_location + 'src/slugs --explicitStrategy --jsonOutput ' + infile + '.slugsin > ' + infile+'.json',shell=True, stdout=subprocess.PIPE)
			sp.wait()
			print('Computing controller...')
			return self.parseJson(infile+'.json')

	def writeJson(self,infile,outfile,dict=None):
		if dict is None:
			dict = self.parseJson(infile)
		j = json.dumps(dict, indent=1)
		f = open(outfile, 'w')
		print >> f, j
		f.close()

	def parseJson(self,filename,outfilename=None):
		automaton = dict()
		file = open(filename)
		data = json.load(file)
		file.close()
		variables = dict()
		for var in data['variables']:
			v = var.split('@')[0]
			if v not in variables.keys():
				for var2ind in range(data['variables'].index(var),len(data['variables'])):
					var2 = data['variables'][var2ind]
					if v != var2.split('@')[0]:
						variables[v] = [data['variables'].index(var), data['variables'].index(var2)]
						break
					if data['variables'].index(var2) == len(data['variables'])-1:
						variables[v] = [data['variables'].index(var), data['variables'].index(var2)+1]

		for s in data['nodes'].keys():
			automaton[int(s)] = dict.fromkeys(['State','Successors'])
			automaton[int(s)]['State'] = dict()
			automaton[int(s)]['Successors'] = []
			for v in variables.keys():
				if variables[v][0] == variables[v][1]:
					bin  = [data['nodes'][s]['state'][variables[v][0]]]
				else:
					bin = data['nodes'][s]['state'][variables[v][0]:variables[v][1]]
				automaton[int(s)]['State'][v] = int(''.join(str(e) for e in bin)[::-1], 2)
				automaton[int(s)]['Successors'] = data['nodes'][s]['trans']
		if outfilename==None:
			return automaton
		else:
			self.writeJson(None,outfilename,automaton)

	def definePolicyDict(self,id_list,policy_array):
		self.policy_list = dict([[i,p_i] for i,p_i in zip(id_list,policy_array)])

	def savePolicy(self):
		print(self.id_no)
		with open('policies/'+str(self.id_no)+'.pkl','wb') as f:
			pickle.dump(self.policy,f,pickle.HIGHEST_PROTOCOL)

	def loadPolicy(self):
		print(self.id_no)
		with open('policies/'+str(self.id_no)+'.pkl','rb') as f:
			self.policy = pickle.load(f)

	def updateAgent(self,state):
		self.current = state
		
	### Belief Rules
	def initBelief(self,agent_id,no_bad,no_targets):
		self.no_bad = no_bad
		no_agents = len(agent_id)
		## Combinarotic approach
		# no_system_states = 0
		# for b_d in range(no_bad+1):
		# 	no_system_states += comb(no_agents,b_d)
		# ## Unknown number of bad agents
		no_system_states = 2**no_targets
		belief_value = 1.0/no_system_states
		base_list =[0,1] # 0 is bad, 1 is good
		total_list = [base_list for n in range(no_targets)]
		self.id_idx = dict([[k,j] for j,k in enumerate(agent_id)])
		self.local_belief = {}
		self.belief_bad = []
		self.belief_bad = []
		for t_p in itertools.product(*total_list):
			# ## Combinartoic approach
			# if sum(t_p) >= no_agents-no_bad:
			# 	self.local_belief.update({t_p:belief_value})
			# ## Unknown number of bad
			self.local_belief.update({t_p: belief_value})
		self.actual_belief = deepcopy(self.local_belief)
		if abs(sum(self.local_belief.values())-1.0)>1e-6:
			raise ProbablilityNotOne("Sum is "+str(sum(self.local_belief.values())))#,"Sum is "+str(sum(self.local_belief.values())))
		if self.async_flag:
			self.neighbor_set = dict()
			self.neighbor_belief = dict()
			self.resetFlags = dict()
			for t_p in self.local_belief:
				self.neighbor_set[t_p] = set()
				self.neighbor_belief[t_p] = dict()
				self.resetFlags[t_p] = True
				for n_i in agent_id:
					self.neighbor_belief[t_p][n_i] = -1 ## b^a_j(theta)

	def updateBelief(self, viewable_agents,target):
		if not target:
			return
		## Synchronous update rule
		tot_b = 0.0
		self.alpha *= self.burn_rate
		self.belief_bad = []
		for b_i in self.local_belief:
			tot_b += self.likelihood(b_i, viewable_agents,target)*self.local_belief[b_i]
		for b_i in self.local_belief:
			self.local_belief[b_i] = (1-self.alpha)*self.local_belief[b_i] + self.alpha*self.likelihood(b_i, viewable_agents, target)*self.local_belief[b_i]/tot_b
		if abs(sum(self.local_belief.values())-1.0)>1e-6:
			raise ProbablilityNotOne("Sum is "+str(sum(self.local_belief.values())))
		if self.evil:
			for b_i in self.local_belief:
				self.local_belief[b_i] = 0
				self.actual_belief[b_i] = 0
			# self.local_belief[(1,0,1,1,1)] = 1.0
			# self.actual_belief[(1,0,1,1,1)] = 1.0
			random_belief = np.random.rand(len(self.local_belief))
			random_belief /= np.sum(random_belief)
			for b_i,r_b in zip(self.local_belief, random_belief):
				self.local_belief[b_i] = r_b
				self.actual_belief[b_i] = r_b

		#
		# ## TODO -remove
		# for b_i in self.local_belief:
		# 	self.actual_belief[b_i] = self.local_belief[b_i]


	def updateVision(self, state, agent_states):
		self.viewable_agents, viewable_states = self.agent_in_view(state, agent_states.values(), agent_states.keys())
		# viewable_states = [self.observation(agent_states[v_s]) for v_s in self.viewable_agents]
		self.updateBelief(self.viewable_agents, viewable_states)

	def ViewProbability(self, viewable_agents, viewable_states):
		view_prob = []
		for a_i,a_s in zip(viewable_agents,viewable_states):
			obs_states = list(self.gw.observable_states[self.current[0]])
			# Find the mission's planned location for the agent
			policy_prob = []
			bad_prob = []
			for obs_s in obs_states:
				policy_prob.append(self.policy_list[a_i].observation((obs_s,a_s[1]), [self.last_seen[a_i][0]], self.last_seen[a_i][1]))
				bad_prob.append(self.policy_list[a_i].bad_observation((obs_s,a_s[1]), [self.last_seen[a_i][0]], self.last_seen[a_i][1]))
			# Find the most likely location of agent in the mission and then the probability of the observed location based on that position.
			obs_probs = self.observation(obs_states, obs_states[np.argmax(np.asarray(policy_prob))], self.error_prob)
			bad_probs = self.observation(obs_states, obs_states[np.argmax(np.asarray(bad_prob))], self.error_prob)
			# Add it to probability tuple
			if self.bad_model:
				view_prob.append([obs_probs[obs_states.index(a_s[0])],bad_probs[obs_states.index(a_s[0])]])
			else:
				view_prob.append(obs_probs[obs_states.index(a_s[0])]/obs_probs.max())
		return view_prob



	def shareBelief(self,belief_arrays):
		actual_belief = {}
		if len(belief_arrays) >= 2*self.no_bad + 1: ## Case 1
			actual_belief = dict()
			for theta in self.actual_belief:
				self.belief_calls += 1
				sorted_belief = sorted([b_a[theta] for b_a in belief_arrays],reverse=True)
				for f in range(self.no_bad):
					sorted_belief.pop()
					if self.av_flag:
						sorted_belief.pop(-1)
				if self.av_flag:
					actual_belief.update({theta: min(self.local_belief[theta],np.mean(sorted_belief))})
				else:
					sorted_belief.append(self.local_belief[theta])
					actual_belief.update({theta:min(sorted_belief)}) # Minimum
				# actual_belief.update({theta:np.average(sorted_belief)}) #Averaging
		else: # Case 2
			for theta in self.actual_belief:
				actual_belief.update({theta:min(self.actual_belief[theta],self.local_belief[theta])})
				# if self.av_flag:
				# 	actual_belief.update({theta:np.mean([self.actual_belief[theta]]+[self.local_belief[theta]])})
				# else:
		# Normalize
		self.actual_belief = dict([[theta, actual_belief[theta] / sum(actual_belief.values())] for theta in actual_belief])
		if self.evil:
			random_belief = np.random.rand(len(self.local_belief))
			random_belief /= np.sum(random_belief)
			for b_i,r_b in zip(self.local_belief,random_belief):
				self.local_belief[b_i] = r_b
				self.actual_belief[b_i] = r_b
		sys_t = list(self.actual_belief.keys())[np.argmax(list(self.actual_belief.values()))]
		for i,s_i in enumerate(sys_t):
			if s_i == 0:
				self.belief_bad.append(i)

	def ADHT(self,belief_arrays):
		global belief_calls
		actual_belief = {}
		neighbor_set = {}
		for theta in self.actual_belief:
			if self.asyncBeliefUpdate(theta,belief_arrays):
				self.belief_calls += 1
				space = set.union(*[self.neighbor_set[x] for x in self.neighbor_set if x is not theta])
				belief_list = []
				agent_order = []
				for j in space:
					belief_list.append(self.neighbor_belief[theta][j])
					agent_order.append(j)
				belief_list,agent_order = zip(*sorted(zip(belief_list,agent_order)))
				neighbor_set[theta] = set(agent_order[self.no_bad:])
				if self.av_flag:
					if theta == (1,1,1,1,1,1,1,0) and self.id_no==560:
						print(" ")
					actual_belief[theta] = min(self.local_belief[theta],np.mean(list(belief_list)[self.no_bad:-1*self.no_bad]))
				else:
					actual_belief[theta] = min([self.local_belief[theta]]+list(belief_list)[self.no_bad:])
			else:
				actual_belief.update({theta: min(self.actual_belief[theta], self.local_belief[theta])})
			if actual_belief[theta] < 0:
				print('Negative')
		self.actual_belief = dict([[theta, actual_belief[theta] / sum(actual_belief.values())] for theta in actual_belief])
		for n_s in neighbor_set:
			self.neighbor_set[n_s] = neighbor_set[n_s]
		if self.evil:
			random_belief = np.random.rand(len(self.local_belief))
			random_belief /= np.sum(random_belief)
			for b_i, r_b in zip(self.local_belief, random_belief):
				self.local_belief[b_i] = r_b
				self.actual_belief[b_i] = r_b

	def asyncBeliefUpdate(self,belief,belief_arrays):
		if self.resetFlags[belief]:
			self.resetBelief(belief,belief_arrays)
		for j in belief_arrays:
			for t_p in [kj for kj in belief_arrays[j] if kj is not belief]: ## t_p is theta_prime
				if self.neighbor_belief[belief][j] == -1: ##j is in
					self.neighbor_set[t_p].add(j)
			self.neighbor_belief[belief][j] = belief_arrays[j][belief] ## Not sure what line 7 does??
		for t_p in [kj for kj in self.local_belief if kj is not belief]:
			if len(self.neighbor_set[t_p]) < 2*self.no_bad + 1:
				return False
		self.resetFlags[belief] = True
		return True
	
	def resetBelief(self,belief,belief_arrays):
		for j in self.neighbor_belief[belief]:
			if j is not self.id_no:
				self.neighbor_belief[belief][j] = -1
		for k in self.local_belief:
			if k is not belief:
				self.neighbor_set[k] = set([self.id_no])
		self.resetFlags[belief] = False
		
	def likelihood(self,sys_status,viewable_agents,target):
		# Work through each element of the tuple, if is likely then good if its unlikely then bad.
		target_ind = self.targets.index(self.current)
		if sys_status[target_ind] == target:
			return self.target_dict[self.current]
		else:
			return 1-self.target_dict[self.current]


	def observation(self):
		if self.current in self.targets:
			return np.random.choice([0,1],1,p=[1-self.target_dict[self.current],self.target_dict[self.current]])
		return None

	# ##### Vision rules
	def updateVision(self, state, agent_states):
		self.viewable_agents, viewable_states = self.agent_in_view(state, agent_states.values(), agent_states.keys())
		target_value = self.observation()
		self.updateTime()
		self.updateBelief(self.viewable_agents, target_value)
		self.updateLastSeen(self.viewable_agents, viewable_states)

	#
	def agent_in_view(self, state, agent_states, agent_id):
		view_agents = []
		view_states = []
		for a_s, a_i in zip(agent_states, agent_id):
			if isinstance(a_s,list):
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