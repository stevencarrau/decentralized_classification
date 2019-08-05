import numpy as np
import random

class Policy():
	
	def __init__(self,mdp,public_mdp,init,target,lookahead,public_target):
		self.mdp = mdp
		# self.nfa = nfa  # Deterministic transitions -- used for nominal trace
		self.init = init
		self.target = target
		self.public_target = public_target
		self.lookahead = lookahead
		self.policy = self.computePolicy(self.target,mdp)
		if target != public_target:
			self.public_policy = self.computePolicy(self.public_target,public_mdp)
		else:
			self.public_policy = self.policy
		self.mc = self.mdp.construct_MC(self.policy)
		self.public_mc = public_mdp.construct_MC(self.public_policy)
		self.nom_trace = self.nominalTrace(self.init)
		
	def computePolicy(self,targ,mdp):
		T = self.lookahead
		R = dict([(s, a, next_s), 0.0] for s in mdp.states for a in mdp.available(s) for next_s in mdp.post(s, a))
		R.update([(s, a, next_s), 1.0] for s in mdp.states for a in mdp.available(s) for next_s in mdp.post(s, a) if next_s in set(targ))
		V,pol_dict = mdp.E_step_value_iteration(R,set(),set())
		return pol_dict
		
	def updatePolicy(self,targ,mdp):
		T = self.lookahead
		R = dict([(s, a, next_s), 0.0] for s in mdp.states for a in mdp.available(s) for next_s in mdp.post(s, a))
		R.update([(s, a, next_s), 1.0] for s in mdp.states for a in mdp.available(s) for next_s in mdp.post(s, a) if next_s in set(targ))
		V, pol_dict = mdp.T_step_value_iteration(R, T)
		self.policy = pol_dict
		self.mc = mdp.construct_MC(self.policy)
		self.updateNominal(self.init)
	
	def nominalTrace(self,loc):
		T = self.lookahead
		return self.mdp.computeTrace(loc,self.public_policy,T, self.public_target)
	
	def updateNominal(self,loc):
		self.nom_trace = self.nominalTrace(loc)
	
	def changeTarget(self,new_target):
		self.target = new_target
		self.updatePolicy()
		
	def sample(self,state):
		s_l = list(self.policy[state])
		random.shuffle(s_l)
		return s_l[0]
	
	def mcProb(self, init, T):
		s = init
		MC_prob = dict([s_t, 1.0] for s_t in self.mdp.states)
		t = 0
		while t < T:
			MC_prob_up = dict([s_t, 0.0] for s_t in self.mdp.states)
			for s_i in s:
				for z in self.mdp.states:
					MC_prob_up[z] += MC_prob[s_i] * self.public_mc[(s_i, z)]
			MC_prob = MC_prob_up  # /sum([MC_prob_up[e] for e in MC_prob_up])
			# assert sum([MC_prob[e] for e in MC_prob])==1.0
			s = [s_e for s_e in self.mdp.states if MC_prob[s_e] != 0.0]
			t += 1
		return MC_prob
	
	def observation(self, est_loc, last_sight, t):
		MC_prob = self.mcProb(last_sight, t)
		return MC_prob[est_loc]