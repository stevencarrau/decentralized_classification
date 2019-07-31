from policy import Policy
from mdp import *
import math
import itertools
from scipy.special import comb
import numpy as np
import random
import operator
from collections import OrderedDict

class ProbablilityNotOne(Exception):
	pass

class Agent():
	
	def __init__(self,init=None,target_list=[],public_list=[],mdp=None,gw_env=None,belief_tracks=None):
		self.id_no = id(self)-1000*math.floor(id(self)/1000)
		self.init = init
		self.current = init
		self.targets = target_list
		self.public_targets = public_list
		self.belief_tracks = belief_tracks
		self.evil = False
		if target_list != public_list:
			self.evil = True
		t_num = list(range(len(target_list)))
		t_list = list(zip(self.targets,t_num))#[t_num[-1]]+t_num[:-1]))
		p_list = list(zip(self.public_targets,t_num))#[t_num[-1]]+t_num[:-1]))
		self.gw = gw_env
		self.viewable_agents = []
		self.last_seen = {}
		self.mdp = mdp
		self.public_mdp = mdp
		labels = dict([])
		pub_labels = dict([])
		for q,s in enumerate(self.targets):
			labels[s] = q
		for q, s in enumerate(self.public_targets):
			pub_labels[s] = q
		self.mdp.add_init(init)
		self.mdp.add_labels(labels)
		# self.nfa = nfa
		# self.public_nfa = nfa
		# self.nfa.add_init(init)
		# self.nfa.add_labels(labels)
		dra = DRA(0,range(len(self.targets)))
		for q in range(len(self.targets)):
			for i in range(len(self.targets)):
				if q == i:
					if q==len(self.targets)-1:
						dra.add_transition(q,q,0)
					else:
						dra.add_transition(q,q,q+1)
				else:
					dra.add_transition(i, q, q)
		self.public_mdp.add_init(init)
		# self.public_nfa.add_init(init)
		self.pmdp = self.productMDP(self.mdp,dra)
		self.public_pmdp = self.productMDP(self.public_mdp,dra)
		# self.pnfa = self.productMDP(self.nfa,dra)
		# self.public_pnfa = self.productMDP(self.public_nfa,dra)
		self.policy = Policy(self.pmdp,self.public_pmdp,self.pmdp.init,t_list,50,p_list)
	
	def writeOutputTimeStamp(self,init=[]):
		out_dict = dict()
		out_dict.update({'AgentLoc': self.current})
		out_dict.update({'ActBelief': self.actual_belief})
		out_dict.update({'LastSeen': self.last_seen})
		out_dict.update({'Visible': self.viewable_agents})
		out_dict.update({'NominalTrace': self.policy.nom_trace})
		if init:
			out_dict.update({'PublicTargets': self.public_targets})
			out_dict.update({'Id_no': list(init)})
			out_dict.update({'BadBelief':self.belief_tracks[0]})
			out_dict.update({'GoodBelief':self.belief_tracks[1]})
		return out_dict
	
	def productMDP(self,mdp,dra):
		init = self.init
		states = []

		for s in mdp.states:
			for q,t_i in enumerate(self.targets):
				states.append((s, q))
		N = len(states)
		labels = dict([])
		trans = []
		for a in mdp.alphabet:
			for i in range(N):
				(s, q) = states[i]
				if type(mdp.L.get(s)) is int:
					labels.update({(s,q):mdp.L[s]})
				for j in range(N):
					(next_s, next_q) = states[j]
					p = mdp.get_prob((s,a,next_s))
					if type(mdp.L.get(s)) is int:
						if next_q == dra.get_transition(mdp.L.get(s),q) and p:
							trans.append(((s,q),a,(next_s,next_q),p))
					elif p and q==next_q:
						trans.append(((s,q),a,(next_s,q),p))
		return MDP(states=list(states),alphabet=mdp.alphabet,transitions=trans,init=init,L=labels)
	
	def definePolicyDict(self,id_list,policy_array):
		self.policy_list = dict([[i,p_i] for i,p_i in zip(id_list,policy_array)])
	
	def agent_in_view(self,state,agent_states,agent_id,error_prob):
		view_agents = []
		view_states = []
		for a_s,a_i in zip(agent_states,agent_id):
			if len(a_s)>1:
				if a_s[0] in self.gw.observable_states[state[0]]:
					view_agents.append(a_i)
					if np.random.rand() < error_prob:
						view_states.append((random.choice(tuple(self.gw.observable_states[state[0]])),a_s[1]))
					else:
						view_states.append(a_s)
			else:
				if a_s in self.gw.observable_states[state]:
					view_agents.append(a_i)
					if np.random.rand() < error_prob:
						view_states.append(random.choice(tuple(self.gw.observable_states[state[0]])))
					else:
						view_states.append(a_s)
		return view_agents,view_states
	
	
	def updateAgent(self,state):
		self.current = state
		
	##### Vision rules
	def updateVision(self,state,agent_states,error_prob):
		self.viewable_agents,viewable_states = self.agent_in_view(state,agent_states.values(),agent_states.keys(),error_prob)
		# viewable_states = [self.observation(agent_states[v_s]) for v_s in self.viewable_agents]
		self.updateTime()
		self.updateBelief(self.viewable_agents,viewable_states,error_prob)
		self.updateLastSeen(self.viewable_agents,viewable_states)
		
	def initLastSeen(self,agent_id,agent_states):
		for a_s,a_i in zip(agent_states,agent_id):
			self.last_seen.update({a_i:[a_s,0]})  # dictionary of lists: [state,time since observed in that state]
			
	def updateLastSeen(self,agent_id,agent_states):
		assert len(agent_id)==len(agent_states)
		for a_i,a_s in zip(agent_id,agent_states):
			self.last_seen[a_i] = [a_s,0]
		
	def updateTime(self):
		for l_i in self.last_seen:
			self.last_seen[l_i][1] += 1

	### Belief Rules
	def initBelief(self,agent_id,no_bad):
		self.no_bad = no_bad
		no_agents = len(agent_id)
		## Combinarotic approach
		no_system_states = 0
		for b_d in range(no_bad+1):
			no_system_states += comb(no_agents,b_d)
		# ## Unknown number of bad agents
		# no_system_states = 2**no_agents
		belief_value = 1.0/no_system_states
		base_list =[0,1] # 0 is bad, 1 is good
		total_list = [base_list for n in range(len(agent_id))]
		self.id_idx = dict([[k,j] for j,k in enumerate(agent_id)])
		self.local_belief = {}
		self.belief_bad = []
		for t_p in itertools.product(*total_list):
			## Combinartoic approach
			if sum(t_p) >= no_agents-no_bad:
				self.local_belief.update({t_p:belief_value})
			## Unknown number of bad
			# self.local_belief.update({t_p: belief_value})
		self.actual_belief = self.local_belief.copy()
		if abs(sum(self.local_belief.values())-1.0)>1e-6:
			raise ProbablilityNotOne("Sum is "+str(sum(self.local_belief.values())))#,"Sum is "+str(sum(self.local_belief.values())))
	
	def updateBelief(self,viewable_agents,viewable_states,error_prob):
		tot_b = 0.0
		self.belief_bad = []
		for b_i in self.local_belief:
			tot_b += self.likelihood(b_i,viewable_agents,viewable_states,error_prob)*self.local_belief[b_i]
		for b_i in self.local_belief:
			self.local_belief[b_i] = self.likelihood(b_i,viewable_agents,viewable_states,error_prob)*self.local_belief[b_i]/tot_b
		if abs(sum(self.local_belief.values())-1.0)>1e-6:
			raise ProbablilityNotOne("Sum is "+str(sum(self.local_belief.values())))
		if self.evil:
			random_belief = np.random.rand(len(self.local_belief))
			random_belief /= np.sum(random_belief)
			for b_i,r_b in zip(self.local_belief,random_belief):
				self.local_belief[b_i] = r_b
				self.actual_belief[b_i] = r_b
	
	def shareBelief(self,belief_arrays):
		actual_belief = {}
		if len(belief_arrays) >= 2*self.no_bad + 1: ## Case 1
			actual_belief = dict()
			for theta in self.actual_belief:
				sorted_belief = sorted([b_a[theta] for b_a in belief_arrays],reverse=True)
				for f in range(self.no_bad):
					sorted_belief.pop()
				sorted_belief.append(self.local_belief[theta])
				actual_belief.update({theta:min(sorted_belief)})
		else: # Case 2
			for theta in self.actual_belief:
				actual_belief.update({theta:min(self.actual_belief[theta],self.local_belief[theta])})
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
	
	def asyncBeliefUpdate(self,belief,belief_arrays,t):
		if t is 0 or self.resetFlag:
			self.resetBelief(belief,belief_arrays)
		for j in belief_arrays:
			for theta in [kj for kj in j if kj is not belief]:
				if belief_arrays[j][theta] == -1: ##j is in
					belief_arrays += j
					
		for theta in [ta for ta in self.actual_belief.keys() if ta is not belief]:
			if len(belief_arrays) < 2*self.no_bad + 1:
				return False
		self.resetFlag = True
		return True
	
	def resetBelief(self,belief,belief_arrays):
		for j in belief_arrays:
			if j is not self:
				j.actual_belief[belief] = -1
		self.resetFlag = False
		
	def likelihood(self,sys_status,viewable_agents,viewable_states,error_prob):
		epsilon = 1e-9
		view_prob = []
		for a_i,a_s in zip(viewable_agents,viewable_states):
			if self.policy_list[a_i].observation(a_s, [self.last_seen[a_i][0]], self.last_seen[a_i][1]) == 1:
				view_prob.append(1.0)
			else:
				view_prob.append(error_prob/len(self.gw.observable_states[self.last_seen[a_i][0][0]]))
		view_index = [self.id_idx[v_a] for v_a in viewable_agents]
		prob_i = 1.0
		for v_i,v_p in zip(view_index,view_prob):
			if sys_status[v_i] == 0: # if bad
				prob_i *= 1.0-v_p+epsilon
			else:
				prob_i *= v_p+epsilon
		return prob_i