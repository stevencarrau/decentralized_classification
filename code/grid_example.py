from gridworld import *
from mdp import *
from policy import Policy
from agent import Agent
import json
import itertools
import pickle
import copy
import itertools



def play_sim(multicolor=True, agent_array=None,grid=None,tot_t=100):
	# Define agent location
	agent_loc = dict([[a.id_no, a.current] for a in agent_array])
	agent_loc_new = copy.deepcopy(agent_loc)
	## Labels for output plots
	plotting_dictionary = dict()
	time_p = {}
	time_t = 0
	for a_a in agent_array:
		time_p.update({a_a.id_no: a_a.writeOutputTimeStamp(agent_loc.keys())})
	plotting_dictionary.update({str(time_t): time_p})

	# Movement loop
	while time_t<tot_t:
		print("Time: "+str(time_t))
		time_p = {}
		time_t += 1
		## Movement update -- Jesse Start HERE
		for ind,a_i in enumerate(agent_array):
			# Loop through agents to update
			agent_loc_new[a_i.id_no] = a_i.update(agent_loc)
		# Update agent location
		agent_loc = agent_loc_new

		# ## Sharing update -- Jesse not needed for you
		packet_dict = {}
		for a_i, p_i in enumerate(agent_array):
			belief_packet,info_packet = beliefPacketFn(p_i,agent_array) ## ADHT belief packet
			packet_dict.update({p_i:(belief_packet,info_packet)})
		for a_i,p_i in enumerate(agent_array):
			belief_packet,info_packet = packet_dict[p_i]
			p_i.ADHT(belief_packet, info_packet)
			time_p.update({p_i.id_no: p_i.writeOutputTimeStamp()})
			# current_env[a_i] = tuple(p_i.comms_env)
		# Plotting dictionary
		plotting_dictionary.update({str(time_t): time_p})

	# Writing outputs
	print("Writing to "+fname)
	write_JSON(fname+'.json', stringify_keys(plotting_dictionary))
	env_file = open(fname+'.pickle','wb')
	pickle.dump(gwg,env_file)
	env_file.close()
	# x = [print('Agent_{}: Steps {}'.format(a_i.id_no,a_i.steps)) for a_i in agent_array]
	# return print("Goal!")


def beliefPacketFn(agent_p,agent_array):
	belief_packet = {}
	for b_a in agent_p.actual_belief:
		belief_dict = {}
		for v_a in agent_p.viewable_agents:
			agent_b = agent_array[agent_p.id_idx[v_a]]
			view_dict = {}
			# for b_z in agent_b.diff_belief[b_a]:
			for b_z in agent_b.actual_belief:
				view_dict.update({b_z:agent_b.actual_belief[b_z]})
			belief_dict.update({v_a:view_dict})
		belief_packet.update({b_a:belief_dict})
	info_packet = {}
	for v_i in agent_array:
		info_packet.update({v_i.id_no:v_i.information})
	return belief_packet,info_packet


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

# Convert co-ordinates to grid cells
def coords(s,ncols):
	return (int(s /ncols), int(s % ncols))


## Define model as a gridworld function
target_prob = 0.8

# # # Specific Scenario -- 5 agents, 3 targets, 1 meeting place
# initial = random.sample(valid_states,no_agents)
initial = [272,470,720,1137,1202,1335,1396,1773,1307,2322,1059,1410]
no_agents = len(initial)
meeting_state = [1223]
targets = [dict([[24,1-target_prob],[1249,target_prob],[2474,target_prob]])]*no_agents
# targets = [dict([[1614,1-target_prob],[884,target_prob]])]*no_agents
no_targets = len(targets[0])
obs_range = 1
np.random.seed(1)

evil_switch = True

obstacles = []
obstacles += list(itertools.chain(*[list(range(i,2500,50)) for i in range(70,74)]))
obstacles += list(range(1000,1450))
obstacles += list(range(20,50))
obstacles += list(range(2470,2500))
obstacles += list(range(49,2500,50))
obstacles = list(set(list(range(2500)))-set(obstacles))
slugs_location = '~/slugs/'
filename = [None]
gwg = Gridworld(initial,nrows=50, ncols=50, nagents=len(initial), targets=targets,regions=None,obs_range=obs_range,meeting_states=meeting_state,obstacles=list(set(obstacles)),filename=filename)

# Create MDP model
# gwg.render()
# gwg.save('Hallways.png')
states = range(gwg.nstates)
alphabet = [0,1,2,3] # North, south, west, east
transitions = []
det_trans = []
for s in states:
	for a in alphabet:
		for t in np.nonzero(gwg.prob[gwg.actlist[a]][s])[0]:
			p = gwg.prob[gwg.actlist[a]][s][t]
			transitions.append((s, alphabet.index(a), t, p))
mdp = MDP(states, set(alphabet),transitions)
print("Models built")


bad_b = ()
for i in range(len(initial)):
	bad_b += (0,)
belief_tracks = [str((1,1,1)), str((0,1,1))] # For output plots
# belief_tracks = [str((1,1)), str((0,1))] # For output plots
seed_iter = iter(range(0,5+len(initial)))


## Initialize Agents -- JESSE HERE
print("Computing policies")
# run_type = 'no_meeting/'
# data_source = 'data/sandia_run/'
agent_array = []
c_i = 0
for i, j in zip(initial, targets):
	np.random.seed(next(seed_iter))
	# file_in = open(data_source+run_type+'platform{}_sTraj'.format(c_i),'r')
	# trace_in = file_in.read().split()
	# trace_san = [int(trace_in[0])]
	# trace_san = trace_in if len(trace_in)<1000 else trace_in[0:-1:416]
	# for t_i in trace_in[1:]:
	# 	if int(t_i) != trace_san[-1]:
	# 		trace_san.append(int(t_i))
	# trace_san = [t_i if t_i!=trace_san[-1] for t_i in trace_in]
	if c_i ==3:
		agent_array.append(Agent(init=i, target_list=j,meeting_state=meeting_state, gw_env=gwg, belief_tracks=belief_tracks, id_no=c_i,
								 policy_load=False, slugs_location=slugs_location, evil=(1,1,1),trace_load=None))
	else:
		agent_array.append(Agent(init=i, target_list=j,meeting_state=meeting_state, gw_env = gwg, belief_tracks=belief_tracks,id_no=c_i,policy_load = False,slugs_location=slugs_location,evil=False,trace_load=None))
	print("Policy ", c_i, " -- complete")
	c_i += 1
id_list = [a_l.id_no for a_l in agent_array] # List of id_nos : not required
agent_loc = dict([[a.id_no, a.current] for a in agent_array]) # Dictionary {agent_id:agent_loc}
# Initialize agent belief and information-sharing structures
for a_i in agent_array:
	a_i.initBelief([a_l.id_no for a_l in agent_array],1,no_targets)
	a_i.initPolicy(pre_load=False,meeting=True)
	a_i.initInfo(agent_loc)

# Run simulation
fname = str('Hallway_Sim_{}_Agents_{}').format(len(agent_array),'Meet')
play_sim(True,agent_array,gwg,300)


