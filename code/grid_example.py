from gridworld import *
from mdp import *
from policy import Policy
from agent import Agent
import json
import itertools
import pickle
import pathlib
import copy

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
	fname = str('Sandia_Sim_{}_Agents_Meet').format(len(agent_array))
	print("Writing to "+fname)
	write_JSON(fname+'.json', stringify_keys(plotting_dictionary))
	env_file = open(fname+'.pickle','wb')
	pickle.dump(gwg,env_file)
	env_file.close()
	x = [print('Agent_{}: Steps {}'.format(a_i.id_no,a_i.steps)) for a_i in agent_array]
	return print("Goal!")


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
random.seed(0)
nrows = 25
ncols = 25
border_size = 10
# no_agents = 5
allowed_region =  list(range(border_size)) + list(range(ncols-border_size,ncols))
moveobstacles = []
valid_states = [o_i for o_i in range(nrows*ncols) if coords(o_i,ncols)[0] not in allowed_region or coords(o_i,ncols)[1] not in allowed_region ]
obstacles = list(set(range(nrows*ncols))-set(valid_states))
target_prob = 0.9

# # # Specific Scenario -- 5 agents, 3 targets, 1 meeting place
# initial = random.sample(valid_states,no_agents)
initial = [86,500,1629,1741,681]
no_agents = len(initial)
meeting_state = [1059]
targets = [dict([[47,1-target_prob],[261,target_prob],[979,target_prob]])]*no_agents
no_targets = len(targets[0])
obs_range = 10
np.random.seed(1)

evil_switch = True

regionkeys = {'pavement','gravel','grass','sand','deterministic'}
regions = dict.fromkeys(regionkeys,{-1})
regions['deterministic']= range(nrows*ncols)
# regions_det = dict.fromkeys(regionkeys,{-1})
# regions_det['deterministic'] = range(nrows*ncols)
slugs_location = '~/slugs/'
env_loc = pathlib.Path().absolute()
# mapname = 'RVR_2_7_20_site_cropped'
# scale = (74,29)

# Create and load map
mapname = 'Sandia'
scale = (49,37)
gwg = Gridworld(initial, nrows, ncols, len(initial), targets, obstacles,moveobstacles,regions=None,obs_range=obs_range,filename=[str(env_loc)+'/'+mapname+'.png',scale,cv2.INTER_LINEAR_EXACT],meeting_states=meeting_state)
# Create MDP model
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
seed_iter = iter(range(0,5+len(initial)))


## Initialize Agents -- JESSE HERE
print("Computing policies")
agent_array = []
c_i = 0
for i, j in zip(initial, targets):
	np.random.seed(next(seed_iter))
	if c_i ==3:
		agent_array.append(Agent(init=i, target_list=j,meeting_state=meeting_state, gw_env=gwg, belief_tracks=belief_tracks, id_no=np.random.randint(1000),
								 policy_load=False, slugs_location=slugs_location, evil=(1,1,0)))
	else:
		agent_array.append(Agent(init=i, target_list=j,meeting_state=meeting_state, gw_env = gwg, belief_tracks=belief_tracks,id_no=np.random.randint(1000),policy_load = False,slugs_location=slugs_location,evil=False))
	print("Policy ", c_i, " -- complete")
	c_i += 1
id_list = [a_l.id_no for a_l in agent_array] # List of id_nos : not required
agent_loc = dict([[a.id_no, a.current] for a in agent_array]) # Dictionary {agent_id:agent_loc}
# Initialize agent belief and information-sharing structures
for a_i in agent_array:
	a_i.initBelief([a_l.id_no for a_l in agent_array],1,no_targets)
	a_i.initInfo(agent_loc)

# Run simulation
play_sim(True,agent_array,gwg,1500)


