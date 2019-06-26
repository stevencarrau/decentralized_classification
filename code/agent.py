from policy import Policy
from mdp import *
import math

class Agent():
	
	def __init__(self,init=None,target_list=[],public_list=[],mdp=None,nfa=None,gw_env=None):
		self.id_no = id(self)-1000*math.floor(id(self)/1000)
		self.init = init
		self.current = init
		self.targets = target_list
		self.public_targets = public_list
		t_num = list(range(len(target_list)))
		t_list = list(zip(self.targets,t_num))#[t_num[-1]]+t_num[:-1]))
		p_list = list(zip(self.public_targets,t_num))#[t_num[-1]]+t_num[:-1]))
		self.gw = gw_env
		self.viewable_agents = []
		self.last_seen = {}
		self.local_belief = {}
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
		self.public_mdp.add_init(init)
		self.public_nfa.add_init(init)
		self.pmdp = self.productMDP(self.mdp,dra)
		self.public_pmdp = self.productMDP(self.public_mdp,dra)
		self.pnfa = self.productMDP(self.nfa,dra)
		self.public_pnfa = self.productMDP(self.public_nfa,dra)
		self.policy = Policy(self.pmdp,self.public_pmdp,self.pnfa,self.public_pnfa,self.pmdp.init,t_list,40,p_list)
	
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
	
	def agent_in_view(self,state,agent_states,agent_id):
		view_agents = []
		for a_s,a_i in zip(agent_states,agent_id):
			if len(a_s)>1:
				if a_s[0] in self.gw.observable_states[state[0]]:
					view_agents.append(a_i)
			else:
				if a_s in self.gw.observable_states[state]:
					view_agents.append(a_i)
		return view_agents
	
	
	def updateAgent(self,state):
		self.current = state
		
	##### Vision rules
	def updateVision(self,state,agent_states):
		self.viewable_agents = self.agent_in_view(state,agent_states.values(),agent_states.keys())
		viewable_states = [agent_states[v_s] for v_s in self.viewable_agents]
		self.updateTime()
		self.updateBelief(self.viewable_agents,viewable_states)
		self.updateLastSeen(self.viewable_agents,viewable_states)
		
	def initLastSeen(self,agent_id,agent_states):
		for a_s,a_i in zip(agent_states,agent_id):
			self.last_seen.update({a_i:[a_s,0]})  # dictionary of lists: [state,time since observed in that state]
			self.local_belief.update({a_i:self.policy.observation(a_s,[a_s],0)})
			
	def updateLastSeen(self,agent_id,agent_states):
		assert len(agent_id)==len(agent_states)
		for a_i,a_s in zip(agent_id,agent_states):
			self.last_seen[a_i] = [a_s,0]
		
	def updateTime(self):
		for l_i in self.last_seen:
			self.last_seen[l_i][1] += 1

	### Belief Rules
	def updateBelief(self,viewable_agents,viewable_states):
		for a_i,a_s in zip(viewable_agents,viewable_states):
			self.local_belief[a_i] = self.policy.observation(a_s,[self.last_seen[a_i][0]],self.last_seen[a_i][1])
	
	