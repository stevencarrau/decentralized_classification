from gridworld import *
from mdp import *
from policy import Policy
from agent import Agent
import json
import pickle

def play_sim(multicolor=True, agent_array=None,grid=None,tot_t=100):
	current_node = [0]*len(agent_array)
	state_label = 's'
	plotting_dictionary = dict()
	time_t = 0
	agent_loc = dict([[a.id_no, a.current] for a in agent_array])
	time_p = {}
	# Initialize missing status
	for a_a in agent_array:
		# a_a.initLastSeen(agent_loc.keys(), agent_loc.values())
		time_p.update({a_a.id_no: a_a.writeOutputTimeStamp(agent_loc.keys())})
	plotting_dictionary.update({str(time_t): time_p})
	target_union = set()
	for t in grid.targets:
		target_union.update(set(t))
	while time_t<tot_t:
		print("Time: "+str(time_t))
		time_p = {}
		time_t += 1
		## Movement update
		for ind,a_i in enumerate(agent_array):
			current_node[ind] = a_i.policy[current_node[ind]]['Successors'][0]
			s = a_i.policy[current_node[ind]]['State'][state_label]
			a_i.updateAgent(s)
			agent_loc[a_i.id_no] = a_i.current
			# print("Local likelihood for ", a_i.id_no, ": ",
			#           a_i.policy.observation((s, q), [prev_state], 1))
		# ## Local update
		for a_i,  p_i in enumerate(agent_array):
			p_i.updateVision(p_i.current, agent_loc)
		# ## Sharing update
		# for a_i, p_i in enumerate(agent_array):
		# 	if p_i.async_flag:
		# 		belief_packet = dict([[v_a,agent_array[p_i.id_idx[v_a]].actual_belief] for v_a in p_i.viewable_agents])
		# 		p_i.ADHT(belief_packet)
		# 	else:
		# 		belief_packet = [agent_array[p_i.id_idx[v_a]].actual_belief for v_a in p_i.viewable_agents]
		# 		p_i.shareBelief(belief_packet)
			time_p.update({p_i.id_no: p_i.writeOutputTimeStamp()})
		plotting_dictionary.update({str(time_t): time_p})
	fname = str('Fixed_Env_{}_Agents.json').format(len(agent_array))
	print("Writing to "+fname)
	write_JSON(fname, stringify_keys(plotting_dictionary))
	return print("Goal!")

def write_JSON(filename,data):
	with open(filename,'w') as outfile:
		json.dump(stringify_keys(data), outfile)

def stringify_keys(d):
	"""Convert a dict's keys to strings if they are not."""
	for key in d.keys():

		# check inner dict
		if isinstance(d[key], dict):
			value = stringify_keys(d[key])
		else:
			value = d[key]

		# convert nonstring to string if needed
		if not isinstance(key, str):
			try:
				d[str(key)] = value
			except Exception:
				try:
					d[repr(key)] = value
				except Exception:
					raise

			# delete old key
			del d[key]
	return d

nrows = 10
ncols = 10
moveobstacles = []
obstacles = []
target_prob = 0.8
# # # 5 agents small range
initial = [33,41,7,80,69]
targets = [dict([[15,1-target_prob],[82,target_prob],[88,target_prob]])]*5
no_targets = len(targets[0])
obs_range = 4
np.random.seed(1)


# # # 6 agents small range
# initial = [(33,0),(41,0),(7,0),(80,0),(69,1),(92,0)]
# targets = [[0,9],[60,69],[20,39],[69,95],[99,11],[9,91]]
# public_targets = [[0,9],[60,69],[20,39],[55,95],[99,11],[9,91]]
# obs_range = 2

# # # 8 agents small range
# initial = [(50,0),(43,0),(75,0),(88,0),(13,0),(37,0),(57,0),(73,0)]
# targets = [[0,90],[3,93],[5,95],[98,8],[9,40],[31,39],[51,59],[55,71]]
# public_targets = [[0,90],[3,93],[5,95],[98,8],[11,19],[31,39],[51,59],[79,71]]
# obs_range = 3

# # 8 agents small range
# initial = [(50,0),(43,0),(75,0),(88,0),(13,0),(37,0),(57,0),(73,0)]
# targets = [[0,90],[3,93],[5,95],[98,8],[9,40],[31,39],[51,59],[55,71]]
# public_targets = [[0,90],[3,93],[5,95],[98,8],[11,19],[31,39],[51,59],[79,53]]
# bad_models = [[8,10],[3,38],[10,85],[91,33],[11,19],[62,68],[23,57],[55,71]]
# obs_range = 4

# ## 8 Spokes on a wheel
# initial = [(0,0),(9,0),(90,0),(99,0),(5,0),(40,0),(94,0),(59,0)]
# public_targets = [[33,0],[36,9],[63,90],[66,99],[54,5],[55,4],[45,94],[44,59]]
# bad_models = [[18,88],[11,91],[36,9],[2,82],[69,99],[27,77],[4,30],[64,59]]
# targets = [[33,0],[36,9],[63,90],[66,99],[54,5],[55,4],[45,94],[64,59]]
# obs_range = 3
# np.random.seed(1)

# ## 6 Spokes on a wheel
# initial = [(0,0),(9,0),(90,0),(99,0),(5,0),(40,0)]
# public_targets = [[33,0],[36,9],[63,90],[66,99],[54,5],[45,94]]
# bad_models = [[18,88],[11,91],[36,9],[2,82],[69,99],[96,59]]
# targets = [[33,0],[36,9],[63,90],[66,99],[54,5],[96,59]]
# obs_range = 4
#
# ## 6 Spokes on a wheel - Tighter centers
# initial = [(0,0),(9,0),(90,0),(99,0),(5,0),(40,0)]
# public_targets = [[43,0],[46,9],[53,90],[56,99],[64,5],[35,94]]
# bad_models = [[18,88],[11,91],[36,9],[2,82],[69,99],[96,59]]
# targets = [[43,0],[46,9],[53,90],[56,99],[64,5],[96,59]]
# obs_range = 3

## 4 Spokes on a wheel - Tighter centers
# initial = [(1,0),(4,0),(55,0),(35,0)]
# public_targets = [[26,1],[27,4],[32,55],[33,58]]
# bad_models = [[25,0],[28,5],[31,54],[34,58]]
# targets = [[26,1],[27,4],[32,55],[34,58]]
# obs_range = 3
# # obs_range = 8


# #4 agents larger range
# initial = [(33,0),(41,0),(7,0),(80,0)]
# targets = [[0,9],[60,69],[20,39],[69,95]]
# public_targets = [[0,9],[60,69],[20,39],[55,95]]
# obs_range = 5

# #4 agents big range
# initial = [(33,0),(41,0),(7,0),(80,0)]
# targets = [[0,9],[60,69],[20,39],[69,95]]
# public_targets = [[0,9],[60,69],[20,39],[55,95]]
# obs_range = 4
#

evil_switch = True

regionkeys = {'pavement','gravel','grass','sand','deterministic'}
regions = dict.fromkeys(regionkeys,{-1})
regions['deterministic']= range(nrows*ncols)
# regions_det = dict.fromkeys(regionkeys,{-1})
# regions_det['deterministic'] = range(nrows*ncols)
slugs_location = '~/slugs/'
gwg = Gridworld(initial, nrows, ncols, len(initial), targets, obstacles,moveobstacles,regions,obs_range=obs_range)
#
states = range(gwg.nstates)
alphabet = [0,1,2,3] # North, south, west, east
transitions = []
det_trans = []
for s in states:
	for a in alphabet:
		for t in np.nonzero(gwg.prob[gwg.actlist[a]][s])[0]:
			p = gwg.prob[gwg.actlist[a]][s][t]
			transitions.append((s, alphabet.index(a), t, p))
		# for t2 in np.nonzero(det_gw.prob[det_gw.actlist[a]][s])[0]:
		#     p_det = det_gw.prob[det_gw.actlist[a]][s][t2]
		#     det_trans.append((s, alphabet.index(a), t2, p_det))

mdp = MDP(states, set(alphabet),transitions)
# nfa = MDP(states,set(alphabet),det_trans) #deterministic transitions
print("Models built")
agent_array = []
c_i = 0
print("Computing policies")
bad_b = ()
for i in range(len(initial)):
	bad_b += (0,)
# belief_tracks = [str(bad_b), str(tuple([int(i==j) for i, j in zip(targets, public_targets)]))]
seed_iter = iter(range(0,5+len(initial)))
for i, j in zip(initial, targets):
	np.random.seed(next(seed_iter))
	if c_i ==3:
		agent_array.append(Agent(init=i, target_list=j, gw_env=gwg, belief_tracks=None, id_no=np.random.randint(1000),
								 policy_load=False, slugs_location=slugs_location, evil=True))
	else:
		agent_array.append(Agent(init=i, target_list=j, gw_env = gwg, belief_tracks=None,id_no=np.random.randint(1000),policy_load = False,slugs_location=slugs_location,evil=False))
	# else:
	#     agent_array.append(Agent(i, j, k, mdp, gwg, belief_tracks, l,np.random.randint(1000),True))
	print("Policy ", c_i, " -- complete")
	c_i += 1
id_list = [a_l.id_no for a_l in agent_array]
# pol_list = [a_l.policy for a_l in agent_array]
for a_i in agent_array:
	a_i.initBelief([a_l.id_no for a_l in agent_array],1,no_targets)
	# a_i.definePolicyDict(id_list,pol_list)

play_sim(True,agent_array,gwg,250)


