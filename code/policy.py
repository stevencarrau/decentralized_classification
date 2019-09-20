import numpy as np
import random

class Policy():
	
	def __init__(self,mdp,public_mdp,init,target,lookahead,public_target,bad_mdp=[],bad_target=[]):
		self.mdp = mdp
		# self.nfa = nfa  # Deterministic transitions -- used for nominal trace
		self.init = init
		self.target = target
		self.public_target = public_target
		self.lookahead = lookahead
		if target != public_target:
			self.public_policy = self.computePolicy(self.public_target,public_mdp)
			self.policy = self.computePolicy(self.target,mdp)
			self.public_mdp = public_mdp
		else:
			self.public_policy = self.computePolicy(self.public_target,public_mdp)
			self.policy = self.public_policy
			self.public_mdp = mdp
		self.mc = self.mdp.construct_MC(self.policy)
		self.public_mc = public_mdp.construct_MC(self.public_policy)
		self.nom_trace = self.nominalTrace(self.init,self.public_mdp,self.public_policy)
		if bad_target:
			self.bad_target = bad_target
			self.bad_mdp = bad_mdp
			if target != public_target:
				self.bad_pol = self.policy
			else:
				self.bad_pol = self.computePolicy(self.bad_target, self.bad_mdp)
			self.bad_mc = self.bad_mdp.construct_MC(self.bad_pol)
			self.bad_trace = self.nominalTrace(self.init,self.bad_mdp,self.bad_pol)
		
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
	
	def nominalTrace(self,loc,mdp,policy=None):
		T = self.lookahead
		if policy:
			return mdp.computeTrace(loc,policy,T)
		else:
			return mdp.computeTrace(loc,self.public_policy,T)
	def updateNominal(self,loc,mdp=None):
		if mdp:
			self.nom_trace = self.nominalTrace(loc,mdp)
		self.nom_trace = self.nominalTrace(loc,self.public_mdp)
		if self.bad_pol:
			self.bad_trace = self.nominalTrace(loc,self.bad_mdp,self.bad_pol)
	
	def changeTarget(self,new_target):
		self.target = new_target
		self.updatePolicy()
		
	def sample(self,state):
		s_l = list(self.policy[state])
		random.shuffle(s_l)
		return s_l[0]
	
	def mcProb(self, init, T,mc=[]):
		if mc ==[]:
			mc = self.public_mc
		s = init
		MC_prob = dict([s_t, 1.0] for s_t in self.mdp.states)
		t = 0
		while t < T:
			MC_prob_up = dict([s_t, 0.0] for s_t in self.mdp.states)
			for s_i in s:
				for z in self.mdp.states:
					MC_prob_up[z] += MC_prob[s_i] * mc[(s_i, z)]
			MC_prob = MC_prob_up  # /sum([MC_prob_up[e] for e in MC_prob_up])
			# assert sum([MC_prob[e] for e in MC_prob])==1.0
			s = [s_e for s_e in self.mdp.states if MC_prob[s_e] != 0.0]
			t += 1
		return MC_prob
	
	def observation(self, est_loc, last_sight, t):
		MC_prob = self.mcProb(last_sight, t)
		return MC_prob[est_loc]
	
	def bad_observation(self,est_loc,last_sight,t):
		MC_prob = self.mcProb(last_sight, t,self.bad_mc)
		return MC_prob[est_loc]