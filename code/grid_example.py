from gridworld import *
from mdp import *
from policy import Policy
from agent import Agent
import json
import pickle

def play_sim(multicolor=True, agent_array=None,grid=None,tot_t=100):
    plotting_dictionary = dict()
    time_t = 0
    agent_loc = dict([[a.id_no, a.current] for a in agent_array])
    time_p = {}
    # Initialize missing status
    for a_a in agent_array:
        a_a.initLastSeen(agent_loc.keys(), agent_loc.values())
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
        for a_i in agent_array:
            agent_loc[a_i.id_no] = a_i.update()

        ## Local update
        for a_i,  p_i in enumerate(agent_array):
            p_i.updateVision(p_i.current, agent_loc)
        ## Sharing update
        for a_i, p_i in enumerate(agent_array):
            if p_i.async_flag:
                belief_packet = dict([[v_a,agent_array[p_i.id_idx[v_a]].actual_belief] for v_a in p_i.viewable_agents])
                p_i.ADHT(belief_packet)
            else:
                belief_packet = [agent_array[p_i.id_idx[v_a]].actual_belief for v_a in p_i.viewable_agents]
                p_i.shareBelief(belief_packet)
            time_p.update({p_i.id_no: p_i.writeOutputTimeStamp()})
        plotting_dictionary.update({str(time_t): time_p})
    fname = str(len(agent_loc))+'agents_'+str(grid.obs_range)+'-'+str('HV')+'_range_async_min.json'
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
# # # 5 agents small range
initial = [33,41,7,80,69]
targets = [[29,0],[60,69],[20,39],[69,96],[99,11]]
public_targets = [[29,0],[60,69],[20,39],[68,93],[99,11]]
bad_models = [[1,19],[60,58],[30,29],[69,96],[81,18]]
obs_range = 6
#
# initial = [31,93,45,194,636,481,88,116,346,800,525,669]
# targets =        [[871,4],[0,899],[5,874],[25,895],[834,37],[52,812],[876,29],[897,60],[14,885],[360,389],[779,390],[780,329]]
# public_targets = [[871,4],[0,899],[5,874],[25,895],[834,37],[52,812],[876,29],[897,60],[14,885],[360,389],[778,420],[750,299]]
# bad_models =     [[872,4],[31,868],[6,875],[26,894],[835,35],[54,808],[877,59],[898,61],[15,886],[390,359],[778,420],[750,299]]


#

evil_switch = True

regionkeys = {'pavement','gravel','grass','sand','deterministic'}
regions = dict.fromkeys(regionkeys,{-1})
regions['deterministic']= range(nrows*ncols)

gwg = Gridworld(initial, nrows, ncols, len(initial), targets, obstacles,moveobstacles,regions,public_targets=public_targets,obs_range=obs_range)
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

mdp = MDP(states, set(alphabet),transitions)
print("Models built")
agent_array = []
c_i = 0
slugs_location = '~/slugs/'
print("Computing policies")
bad_b = ()
for i in range(len(initial)):
    bad_b += (0,)
belief_tracks = [str(bad_b), str(tuple([int(i==j) for i, j in zip(targets, public_targets)]))]
for i, j, k, l in zip(initial, targets, public_targets, bad_models):
    agent_array.append(Agent(init=i, target_list=j, public_list=k, mdp=mdp, gw_env=gwg, belief_tracks=belief_tracks, bad_models=l,id_no=c_i,slugs_location=slugs_location))
    print("Policy ", c_i, " -- complete")
    c_i += 1
id_list = [a_l.id_no for a_l in agent_array]
pol_list = [a_l.policy for a_l in agent_array]
for a_i in agent_array:
    a_i.initBelief([a_l.id_no for a_l in agent_array],2)
    a_i.definePolicyDict(id_list,pol_list)


fname =  str(len(agent_array))+'agents_'+str(obs_range)+'-'+str('HV')+'_range_async_min'
env_file = open(fname + '.pickle', 'wb')
pickle.dump(gwg, env_file)
env_file.close()
play_sim(True,agent_array,gwg,100)


