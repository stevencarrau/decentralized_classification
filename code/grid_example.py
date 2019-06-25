from gridworld import *
from mdp import *
from policy import Policy
from agent import Agent

# from tqdm import tqdm
nrows = 10
ncols = 10
goodtargets = {0}
initial = [(33,0),(40,0),(7,0),(80,0)]
moveobstacles = []
targets = [[0,9],[60,69],[20,39],[79,95]]
public_targets = [[0,9],[60,69],[20,39],[55,95]]
obstacles = []

evil_switch = False

regionkeys = {'pavement','gravel','grass','sand','deterministic'}
regions = dict.fromkeys(regionkeys,{-1})
regions['pavement']= range(nrows*ncols)
regions_det = dict.fromkeys(regionkeys,{-1})
regions_det['deterministic'] = range(nrows*ncols)

gwg = Gridworld(initial, nrows, ncols, len(initial), public_targets, obstacles,moveobstacles,regions)
det_gw = Gridworld(initial, nrows, ncols, len(initial), targets, obstacles,moveobstacles,regions_det)
gwg.render(multicolor=True)
# gwg.draw_state_labels()
# gwg.save('Examples/example_7x5.png')
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
        for t2 in np.nonzero(det_gw.prob[det_gw.actlist[a]][s])[0]:
            p_det = det_gw.prob[det_gw.actlist[a]][s][t2]
            det_trans.append((s, alphabet.index(a), t2, p_det))

mdp = MDP(states, set(alphabet),transitions)
nfa = MDP(states,set(alphabet),det_trans) #deterministic transitions
print("Models built")
agent_array = []
c_i = 0
print("Computing policies")
for i,j,k in zip(initial,targets,public_targets):
    if evil_switch:
        agent_array.append(Agent(i,j,k,mdp,nfa))
    else:
        agent_array.append(Agent(i,k, k, mdp, nfa))
    print("Policy ",c_i," -- complete")
    c_i += 1

gwg.play(True,agent_array)
#
