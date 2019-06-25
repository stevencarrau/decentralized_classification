from policy import Policy
from mdp import *

class Agent():
	
	def __init__(self,init=None,target_list=[],public_list=[],mdp=None,nfa=None):
		self.init = init
		self.targets = target_list
		self.public_targets = public_list
		t_num = list(range(len(target_list)))
		t_list = list(zip(self.targets,t_num))#[t_num[-1]]+t_num[:-1]))
		p_list = list(zip(self.public_targets,t_num))#[t_num[-1]]+t_num[:-1]))
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
		self.nfa = nfa
		self.public_nfa = nfa
		self.nfa.add_init(init)
		self.nfa.add_labels(labels)
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
		self.pmdp = self.productMDP(self.mdp,dra)
		self.pnfa = self.productMDP(self.nfa,dra)
		self.policy = Policy(self.pmdp,self.pnfa,self.pmdp.init,t_list,40,p_list)
	
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