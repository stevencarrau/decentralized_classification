from mdp import *
import json
import pickle
import itertools
import random
import json_writer

event_triggers = {'nominal':1,'ice':6,'alarm':4,'bang':2}
event_names = {0:'nominal',1:'ice',2:'alarm',3:'bang'}
event_space = list(range(len(event_triggers)+1))
# event_trans = [(0,i) for i in event_space[1:]] + [(i,0) for i in event_space[1:]]

## Hard code environment
# 0 - street, 1 - store A, 2-electricity box, 3 - house,4 - store B
env_states = [0,1,2,3,4]
env_trans = [(0,1),(1,2),(1,1),(1,0),(0,4),(4,4),(4,0),(2,2),(2,1),(0,3),(3,0),(3,3)]
env_tracks = {(0,1):[645,615,585,555,525,524,523,522,521,520,519,518,488,458,428,429],
			  (0,0):[645,645,645,645,645,645,645,645,645,645,645,645,645,645,645,645],
			  (1,2):[399,369,368,338,339,340,341,342,312,282,283,283,283,283,283,283],
			  (1,1):[399,369,368,367,366,396,426,456,457,427,427,429,429,429,429,429],
			  (1,0):list(reversed([645,615,585,555,525,524,523,522,521,520,519,518,488,458,428,429])),
			  (0,4):list(reversed([855,855,855,855,855,855,855,855,855,855,825,795,765,735,705,645])),
			  (4,4):[885,886,886,857,827,826,825,824,823,853,883,884,854,824,825,855],
			  (4,0):[855,855,855,855,855,855,855,855,855,855,825,795,765,735,705,645],
			  (2,2):[284,254,253,254,284,285,255,225,195,194,193,192,222,252,282,283],
			  (2,1):list(reversed([399,369,368,338,339,340,341,342,312,282,283,283,283,283,283,283])),
			  (0,3):[645,615,585,555,525,526,527,528,529,530,531,532,502,472,442,442],
			  (3,0):list(reversed([645,615,585,555,525,526,527,528,529,530,531,532,502,472,442,442])),
			  (3,3):[442,443,444,414,413,412,411,410,380,381,382,383,384,414,413,412]}
# product_states = itertools.product(env_states,event_space)
# product_dict = dict([[s,i]] for i, s in enumerate(product_states))
# product_ind = dict([[i,s]] for i, s in enumerate(product_states))

# Shop A - agent
shop_type = 1
Store_a_trans = [(1,0,1,0.9),(1,0,0,0.05),(1,0,2,0.05),(2,0,1,0.95),(2,0,2,0.05),(3,0,0,1.0),(0,0,1,1.0),(4,0,0,1.0), # nominal
		 (1,1,1,0.8),(1,1,0,0.15),(1,1,2,0.05), (2,1,2,0.05),(2,1,1,0.95),(0,1,0,1.0), # ice-cream
		 (1,2,1,0.05),(1,2,0,0.80),(1,2,2,0.15),(2,2,2,1.0),(0,2,0,1.0), # alarm
		 (1,3,1,0.1),(1,3,0,0.85),(1,3,2,0.05),(2,3,2,1.0),(0,3,0,1.0) # bang
		 ]

shop_a_mdp  = MDP(states=env_states,alphabet=event_space,transitions=Store_a_trans)


Store_b_trans = [(4,0,4,0.9),(4,0,0,0.1),(0,0,4,1.0),(1,0,0,1.0),(3,0,0,1.0), # nominal
		 (4,1,4,0.8),(4,1,0,0.2),(0,1,0,1.0), # ice-cream
		 (4,2,4,0.05),(4,2,0,0.95),(0,2,0,1.0), # alarm
		 (4,3,4,0.1),(4,3,0,0.9),(0,3,0,1.0) # bang
		 ]

shop_b_mdp  = MDP(states=env_states,alphabet=event_space,transitions=Store_b_trans)

repair_trans = [(0,0,1,0.5),(0,0,4,0.5),(1,0,2,0.5),(1,0,0,0.5),(2,0,1,1.0),(4,0,0,1.0), # nominal
		 (2,1,2,0.15),(2,1,1,0.85),(1,1,0,0.85),(1,1,2,0.15),(4,1,0,1.0),(0,1,0,0.85),(0,1,1,0.15), # ice-cream
		 (2,2,1,1.0),(1,2,0,0.9),(1,2,1,0.1),(0,2,0,0.9),(0,2,4,0.1),(4,2,0,0.8),(4,2,4,0.2), # alarm
		 (2,3,1,0.5),(2,3,2,0.5),(1,3,2,0.5),(1,3,1, 0.5),(0,3,1,1.0),(4,3,0,1.0),(3,3,0,1.0) # bang
		 ]
repair_mdp  = MDP(states=env_states,alphabet=event_space,transitions=repair_trans)

shopper_trans = [(0,0,1,0.45),(0,0,4,0.45),(0,0,3,0.1),(1,0,2,0.25),(1,0,0,0.5),(1,0,1,0.25),(2,0,1,0.85),(2,0,2,0.15),(4,0,0,0.8),(4,0,4,0.2),(3,0,0,0.9),(3,0,3,0.1), # nominal
		 (2,1,2,0.05),(2,1,1,0.95),(1,1,0,0.85),(1,1,1,0.1),(1,1,2,0.05),(0,1,0,0.85),(0,1,1,0.15),(4,1,4,0.1),(4,1,0,0.9),(3,1,0,1.0), # ice-cream
		(2,2,2,0.1),(2,2,1,0.9),(1,2,0,0.9),(1,2,2,0.1),(0,2,0,1.0),(4,2,0,1.0),(3,2,0,1.0), #alarm
		(2,3,1,0.85),(2,3,2,0.15),(1,3,2,0.15),(1,3,0,0.85),(0,3,0,1.0),(4,3,0,1.0),(3,3,0,1.0) # bang
		 ]
shopper_mdp  = MDP(states=env_states,alphabet=event_space,transitions=shopper_trans)

threat_trans = [(0,0,1,0.45),(0,0,4,0.45),(0,0,3,0.1),(1,0,2,0.25),(1,0,0,0.5),(1,0,1,0.25),(2,0,1,0.85),(2,0,2,0.15),(4,0,0,0.8),(4,0,4,0.2),(3,0,0,0.9),(3,0,3,0.1), # nominal
		 (2,1,2,0.6),(2,1,1,0.4),(1,1,0,0.15),(1,1,1,0.25),(1,1,2,0.6),(0,1,0,0.15),(0,1,1,0.85),(4,1,4,0.25),(4,1,0,0.75),(3,1,0,1.0), # ice-cream
		(2,2,2,0.9),(2,2,1,0.1),(1,2,0,0.1),(1,2,2,0.9),(0,2,0,0.1),(0,2,1,0.9),(4,2,0,1.0),(3,2,0,1.0), #alarm
		(2,3,2,0.8),(2,3,1,0.2),(1,3,0,0.2),(1,3,2,0.8),(0,3,0,0.2),(0,3,1,0.8),(4,3,0,1.0),(3,3,0,1.0) # bang
		 ]
threat_mdp  = MDP(states=env_states,alphabet=event_space,transitions=threat_trans)

home_trans = [(3,0,3,0.9),(3,0,0,0.1),(0,0,3,0.9),(0,0,1,0.05),(0,0,4,0.05),(4,0,4,0.05),(4,0,0,0.95),(1,0,1,0.05),(1,0,0,0.95), # nominal
		 (3,1,3,0.1),(3,1,0,0.9),(0,1,0,1.0),(1,1,0,1.0),(4,1,0,1.0), # ice-cream
		 (3,2,3,0.2),(3,2,0,0.8),(0,2,0,1.0),(1,2,0,1.0),(4,2,0,1.0), # alarm
		 (3,3,3,0.8),(3,3,0,0.2),(0,3,0,1.0),(1,3,0,1.0),(4,3,0,1.0) # bang
		 ]
home_mdp  = MDP(states=env_states,alphabet=event_space,transitions=home_trans)

agents = [shop_a_mdp,shop_b_mdp,repair_mdp,shopper_mdp,threat_mdp,home_mdp]
agent_state = [1,4,2,0,4,3]
agent_tracks = [[] for a_i in agents]
T = 20

event_active = False
event = 0
act_time = 0
for i in range(T):
	for a_i,a_m in enumerate(agents):
		if not event_active:
			event = random.randint(0,3)
			act_time = 0
			event_active = True
		if act_time >= event_triggers[event_names[event]]:
			event = 0
			event_active = False
		next_s = a_m.sample(agent_state[a_i],event)
		agent_tracks[a_i] += env_tracks[(agent_state[a_i],next_s)]
		agent_state[a_i] = next_s
	act_time += 1

agent_paths = json_writer.all_agent_tracks(list(range(6)), agent_tracks)
json_writer.write_JSON('AgentPaths_MDP.json', agent_paths)

# actions - triggered events
